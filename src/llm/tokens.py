"""Token estimate used until a model-specific tokenizer is wired in stage 6.

ASCII text is counted at about four characters per token. Non-ASCII text,
including Persian, is counted at two characters per token. That over-estimates
Persian slightly so a later budget check fails closed instead of overflowing
the window.
"""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    other = len(text) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + (other + 1) // 2)
