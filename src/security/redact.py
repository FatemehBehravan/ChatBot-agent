"""Hide personal data and instruction-like text before a model sees a tool result."""

import re
from typing import Any

_PHONE = re.compile(r"\+\d{2}-\d{3}-\d{2,}")
_INJECTION = re.compile(r"(?is)(?:system notice|ignore all previous instructions).*")
_MASKED_PHONE = "+00-***-****"


def mask_phones(text: str) -> str:
    return _PHONE.sub(_MASKED_PHONE, text)


def contains_phone(text: str) -> bool:
    return _PHONE.search(text) is not None


def present(
    value: Any,
    *,
    pii: str,
    pii_fields: list[str],
    untrusted_fields: list[str],
) -> Any:
    return _walk(value, None, pii=pii, pii_fields=set(pii_fields), untrusted_fields=set(untrusted_fields))


def _walk(
    value: Any,
    key: str | None,
    *,
    pii: str,
    pii_fields: set[str],
    untrusted_fields: set[str],
) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for child_key, child in value.items():
            if child_key in pii_fields and pii == "none":
                continue
            cleaned[child_key] = _walk(
                child,
                child_key,
                pii=pii,
                pii_fields=pii_fields,
                untrusted_fields=untrusted_fields,
            )
        return cleaned
    if isinstance(value, list):
        return [
            _walk(item, key, pii=pii, pii_fields=pii_fields, untrusted_fields=untrusted_fields)
            for item in value
        ]
    if isinstance(value, str):
        return _text(value, key, pii=pii, pii_fields=pii_fields, untrusted_fields=untrusted_fields)
    return value


def _text(
    value: str,
    key: str | None,
    *,
    pii: str,
    pii_fields: set[str],
    untrusted_fields: set[str],
) -> str:
    if key in pii_fields and pii == "masked":
        if key == "phone":
            return _MASKED_PHONE
        first = value[:1]
        return f"{first}***" if first else "***"
    text = value
    if key in untrusted_fields:
        text = _INJECTION.sub("", text).strip(" .\n")
        text = mask_phones(text)
        return f"[data] {text}" if text else ""
    if pii != "own_full":
        text = mask_phones(text)
    return text
