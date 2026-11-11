# Always-on vLLM v1 logits processor that hard-masks every vocabulary token
# containing a Han (Chinese) ideograph or CJK punctuation/fullwidth glyph.
#
# Loaded via `--logits-processors ban_chinese:BanChineseLogitsProcessor`
# (see entrypoint.sh, gated by BAN_CHINESE). Applies unconditionally to every
# request in the batch -- no per-request opt-in/opt-out, so update_state() is
# a no-op and there is no BatchUpdate bookkeeping to do.
import torch
from vllm.config import VllmConfig
from vllm.tokenizers import get_tokenizer
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

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
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _compute_banned_token_ids(tokenizer) -> list[int]:
    """Bans tokens whose raw bytes fully decode to text containing a banned
    char. Deliberately does NOT ban incomplete-UTF-8 byte-fallback fragment
    tokens (tokens that only spell a real character when chained with
    sibling fragments): those fragments are also shared by legitimate
    non-Han content (emoji, some currency/math symbols), and banning them
    caused a demonstrated side effect (silently dropping an otherwise-valid
    emoji) with no measured benefit against real leaks -- production eval
    data (200 real humanization examples) showed 100% of observed spontaneous
    Chinese leaks used complete single tokens, none required byte-fragment
    assembly. A fragment-based bypass remains theoretically possible under
    deliberate/adversarial prompting, which is out of scope here.
    """
    byte_map = _byte_level_reverse_map()
    banned = []
    for token_str, token_id in tokenizer.get_vocab().items():
        try:
            raw = bytes(byte_map[c] for c in token_str)
        except KeyError:
            continue  # not a byte-level-encoded token (e.g. a literal added/special token)
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue  # incomplete byte-fallback fragment -- not banned, see above
        if _contains_banned_char(decoded):
            banned.append(token_id)
    return banned


class BanChineseLogitsProcessor(LogitsProcessor):
    """Hard-masks (-inf) every token whose decoded text contains a Han
    ideograph or CJK punctuation/fullwidth character. Always on for every
    request; no per-request configuration."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        model_config = vllm_config.model_config
        tokenizer = get_tokenizer(
            model_config.tokenizer or model_config.model,
            trust_remote_code=model_config.trust_remote_code,
        )
        banned_ids = _compute_banned_token_ids(tokenizer)
        print(
            f"[ban_chinese] banned {len(banned_ids)} tokens "
            f"(of {len(tokenizer.get_vocab())} vocab entries) "
            f"containing Han ideographs / CJK punctuation",
            flush=True,
        )
        self._banned_ids = torch.tensor(banned_ids, dtype=torch.long, device=device)

    def is_argmax_invariant(self) -> bool:
        return False  # can and does remove the argmax token

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        pass  # unconditional: no per-request state to track

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits[:, self._banned_ids] = float("-inf")
        return logits
