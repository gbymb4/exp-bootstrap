# Always-on vLLM logits processor that hard-masks every vocabulary token
# containing a Han (Chinese) ideograph or CJK punctuation/fullwidth glyph.
#
# Loaded via `--logits-processors ban_chinese:BanChineseLogitsProcessor`.
# It applies unconditionally to every request in the batch, so there is no
# per-request bookkeeping.
#
# Compatibility: vLLM 0.32 moved the base class from
#   vllm.v1.sample.logits_processor.LogitsProcessor
# to
#   vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor
# and changed the constructor from (vllm_config, device, is_pin_memory) to
# (vllm_config, req_states), and replaced update_state/is_argmax_invariant with
# add_request/apply_staged_writes. This module detects the base at import time
# so one file serves both 0.17.x and 0.32.x.
import torch

try:  # vLLM >= 0.32
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
    _NEW_API = True
except Exception:  # vLLM <= 0.17.x
    from vllm.v1.sample.logits_processor import BatchUpdate as _BatchUpdate
    from vllm.v1.sample.logits_processor import LogitsProcessor as _Base
    _NEW_API = False

from vllm.config import VllmConfig
from vllm.tokenizers import get_tokenizer

# Unicode ranges covering Han ideographs and CJK-specific punctuation/fullwidth
# forms. Deliberately excludes kana/hangul ranges -- see repo discussion for why
# a Han-only ban was chosen over a broader CJK ban.
_HAN_AND_CJK_PUNCT_RANGES = (
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F),  # CJK Unified Ideographs Extension B-F + Compat Supplement
    (0x2E80, 0x2EFF),    # CJK Radicals Supplement
    (0x2F00, 0x2FDF),    # Kangxi Radicals
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
)


def _contains_banned_char(s: str) -> bool:
    return any(
        lo <= ord(ch) <= hi
        for ch in s
        for lo, hi in _HAN_AND_CJK_PUNCT_RANGES
    )


def _byte_level_reverse_map() -> dict[str, int]:
    """Inverse of GPT-2's byte-level encoder (see `bytes_to_unicode` in HF's
    tokenization_gpt2.py): maps each single-char vocab symbol back to the raw
    byte value it represents."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _compute_banned_token_ids(tokenizer) -> list[int]:
    """Token ids whose decoded form contains a Han ideograph or CJK punctuation."""
    reverse = _byte_level_reverse_map()
    banned: list[int] = []
    for token_id in range(len(tokenizer.get_vocab())):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(piece, str):
            continue
        # Byte-level vocab: decode the synthetic symbols back to raw bytes.
        decoded = piece
        if all(ch in reverse for ch in piece):
            try:
                decoded = bytes(reverse[ch] for ch in piece).decode("utf-8", errors="ignore")
            except Exception:
                decoded = piece
        if _contains_banned_char(decoded):
            banned.append(token_id)
    return banned


class BanChineseLogitsProcessor(_Base):
    """Hard-masks (-inf) every token whose decoded text contains a Han
    ideograph or CJK punctuation/fullwidth character. Always on for every
    request; no per-request configuration."""

    def __init__(self, vllm_config: VllmConfig, *args):
        # 0.32 passes req_states; older versions pass (device, is_pin_memory).
        device = None
        if not _NEW_API and args:
            device = args[0]
        model_config = vllm_config.model_config
        tokenizer = get_tokenizer(
            model_config.tokenizer or model_config.model,
            trust_remote_code=model_config.trust_remote_code,
        )
        banned_ids = _compute_banned_token_ids(tokenizer)
        print(
            f"[ban_chinese] banned {len(banned_ids)} tokens "
            f"(of {len(tokenizer.get_vocab())} vocab entries) "
            f"containing Han ideographs / CJK punctuation (vLLM API: "
            f"{'v2' if _NEW_API else 'v1'})",
            flush=True,
        )
        self._banned_ids = torch.tensor(banned_ids, dtype=torch.long, device=device or "cpu")

    def apply(self, logits: torch.Tensor, ctx=None) -> torch.Tensor:
        # vLLM 0.32's v2 sampler calls apply(logits, ctx); older versions call apply(logits).
        ids = self._banned_ids.to(logits.device, non_blocking=True)
        logits[:, ids] = float("-inf")
        return logits

    # --- vLLM 0.32 hooks ---
    # add_request must return a bool: vLLM does `needs_processing |= proc.add_request(...)`
    # (sampler.py). Returning False means "no per-request work", which is correct here
    # because the ban applies unconditionally to every request.
    def add_request(self, req_idx: int, sampling_params) -> bool:
        return False

    def apply_staged_writes(self, *args, **kwargs) -> None:
        pass

    # --- older hooks, kept so 0.17.x stays happy ---
    def is_argmax_invariant(self) -> bool:
        return False  # can and does remove the argmax token

    def update_state(self, batch_update=None) -> None:
        pass
