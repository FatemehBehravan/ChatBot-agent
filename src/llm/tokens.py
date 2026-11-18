"""Token estimate shared by every provider.

This is not a model tokenizer. ASCII text is about four characters per token.
Persian and other non-ASCII text are counted at two characters per token,
which is deliberately high, so a fitted prompt stays inside the window.
"""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    other = len(text) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + (other + 1) // 2)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_per_1m: float | None,
    output_per_1m: float | None,
) -> float:
    """USD for one call. Unpriced models cost nothing, so the token ceiling is the guard."""
    return (input_tokens * (input_per_1m or 0) + output_tokens * (output_per_1m or 0)) / 1_000_000
