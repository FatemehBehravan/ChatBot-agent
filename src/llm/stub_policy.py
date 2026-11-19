"""Deterministic stand-in for a tool-calling model.

The agent loop is the same one a live model uses. This module only decides
the next tool call. Sentences that restate a tool result come from
src.agent.answers, which the live turn uses as well.
"""

import json
import re
import uuid

from src.agent.answers import STATUS_FA as _STATUS_FA
from src.agent.answers import estimate_reply, format_own_orders, own_orders_requested, two_periods
from src.llm.types import ChatMessage, ToolCall

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_ORDER = re.compile(r"A-\d+", re.IGNORECASE)
_DISTANCE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km|kilometers?|کیلومتر)")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_WORD_EN = re.compile(r"[A-Za-z]{5,}")
_WORD_FA = re.compile(r"[\u0600-\u06FF]{4,}")
_STOP = {
    "about",
    "after",
    "again",
    "already",
    "could",
    "exact",
    "please",
    "their",
    "there",
    "which",
    "would",
    "order",
    "orders",
}

OUT_OF_SCOPE_EN = (
    "That is outside delivery support. I can help with orders, delivery estimates, and Alopeyk help pages."
)
OUT_OF_SCOPE_FA = "این درخواست خارج از پشتیبانی ارسال است. می‌توانم دربارهٔ سفارش‌ها، تخمین زمان تحویل و راهنمای الوپیک کمک کنم."
_GREETINGS = {"سلام", "درود", "hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
_SERVICES = (
    ("get_order_status", "وضعیت یک سفارش را با شماره‌اش بگویم", "look up an order from its id"),
    ("list_my_orders", "سفارش‌های خودتان را فهرست کنم", "list your own orders"),
    ("search_orders", "بین سفارش‌ها جستجو کنم", "search orders"),
    ("estimate_delivery_time", "زمان معمول تحویل را برای یک مسیر تخمین بزنم", "estimate a typical delivery time"),
    ("estimate_price_range", "بازهٔ قیمت را از سفارش‌های قبلی بگویم", "give a price range from past orders"),
    ("search_knowledge_base", "راهنمای الوپیک را برای بسته‌بندی و قوانین ارسال پیدا کنم", "find Alopeyk help pages"),
    ("create_support_ticket", "برای یک سفارش تیکت پشتیبانی ثبت کنم", "open a support ticket"),
    (
        "request_order_cancellation",
        "لغو سفارش را پیش از تحویل‌گیری به پیک ثبت کنم",
        "request a cancellation before pickup",
    ),
)
_EXAMPLES = (
    (
        "estimate_delivery_time",
        "مثلاً بپرسید: برای مسیر ۵ کیلومتری با موتور در تهران تحویل چقدر طول می‌کشد؟",
        "For example: how long does a 5 km motorcycle delivery in Tehran usually take?",
    ),
    (
        "estimate_price_range",
        "مثلاً بپرسید: قیمت مسیر ۵ کیلومتری با موتور معمولاً چقدر است؟",
        "For example: what does a 5 km motorcycle delivery usually cost?",
    ),
    (
        "search_knowledge_base",
        "مثلاً بپرسید: اقلام شکستنی را چطور بسته‌بندی کنم؟",
        "For example: how should I pack something fragile?",
    ),
    (
        "list_my_orders",
        "مثلاً بگویید: می‌خواهم سفارش‌هایم را ببینم.",
        "For example: show me my orders.",
    ),
    (
        "get_order_status",
        "مثلاً شمارهٔ سفارش را بفرستید، مثل A-246001.",
        "For example, send an order id such as A-246001.",
    ),
)
CLARIFY_EN = "Which order id should I check?"
CLARIFY_FA = "شمارهٔ کدام سفارش را بررسی کنم؟"


def plan(messages: list[ChatMessage], tools: list[dict] | None) -> tuple[str, list[ToolCall]]:
    available = {item["name"] for item in tools} if tools else set()
    user_text = _last_user(messages)
    persian = bool(_PERSIAN.search(user_text))
    if _observations(messages):
        return _continue(messages, available, persian, user_text)
    return _from_user(messages, available, persian, user_text)


def _from_user(
    messages: list[ChatMessage], available: set[str], persian: bool, user_text: str
) -> tuple[str, list[ToolCall]]:
    folded = user_text.translate(_DIGITS).lower()
    if _is_confirmation(folded):
        previous = _previous_user(messages)
        if previous and _asked_to_confirm(messages):
            return _start_write(previous, available, persian)
        return _say(CLARIFY_FA if persian else "What should I confirm?")

    if _is_greeting(folded):
        return _say(greeting(available, persian))

    if "drop table" in folded or "distance_km" in folded:
        quoted = re.search(r'distance_km\s*=\s*"([^"]*)"', user_text, re.IGNORECASE)
        raw = quoted.group(1) if quoted else "invalid"
        return _maybe_call(
            "estimate_delivery_time",
            {"distance_km": raw, "vehicle_type": "motorcycle"},
            available,
            persian,
        )

    if _asks_phone(folded):
        return _say("شمارهٔ تلفن مشتری را نمی‌توانم بگویم." if persian else "I can't share customer phone numbers.")

    order_id = _order_id(user_text)
    if _wants_ticket(folded) and order_id:
        return _ask_ticket(order_id, persian)
    if _wants_cancel(folded) and order_id:
        return _maybe_call("get_order_status", {"order_id": order_id}, available, persian)

    if order_id and (_wants_status(folded) or not _wants_knowledge(folded)):
        return _maybe_call("get_order_status", {"order_id": order_id}, available, persian)

    if not order_id and _refers_to_an_order(folded):
        prior = _history_order_ids(messages)
        if prior:
            chosen = prior[0] if "first" in folded or "اول" in user_text else prior[-1]
            return _maybe_call("get_order_status", {"order_id": chosen}, available, persian)
        return _say(CLARIFY_FA if persian else CLARIFY_EN)

    if _asks_directory(folded):
        if "search_orders" in available:
            arguments: dict = {"limit": 20}
            city = _city(folded)
            status = _directory_status(folded)
            if city:
                arguments["city"] = city
            if status:
                arguments["status"] = status
            if len(arguments) == 1:
                arguments["status"] = "in_transit"
            return _call("search_orders", arguments)
        return _say(
            "این حساب فقط سفارش‌های خودش را می‌بیند. شمارهٔ سفارش را بفرستید."
            if persian
            else "This account can only see its own orders. Send an order id."
        )

    if not order_id and _wants_own_orders(folded):
        if "list_my_orders" in available:
            return _call("list_my_orders", {"limit": 5})
        return _say(CLARIFY_FA if persian else CLARIFY_EN)

    distance = _distance(folded)
    if distance is not None and _wants_price(folded):
        return _maybe_call("estimate_price_range", _estimate_args(folded, distance, None), available, persian)
    if distance is not None and _wants_eta(folded):
        window = "evening" if _wants_both_periods(folded) else None
        return _maybe_call(
            "estimate_delivery_time", _estimate_args(folded, distance, window), available, persian
        )

    if _wants_knowledge(folded):
        return _maybe_call("search_knowledge_base", {"query": user_text, "limit": 3}, available, persian)

    return _say(OUT_OF_SCOPE_FA if persian else OUT_OF_SCOPE_EN)


def _continue(
    messages: list[ChatMessage], available: set[str], persian: bool, user_text: str
) -> tuple[str, list[ToolCall]]:
    done = _turn_results(messages)
    if not done:
        return _from_user(messages, available, persian, user_text)
    name, arguments, result = done[-1]
    folded = user_text.translate(_DIGITS).lower()

    if result.get("status") == "confirmation_required" and name in available:
        follow = dict(arguments)
        follow["confirmed"] = True
        follow["confirmation_id"] = result.get("confirmation_id")
        return _call(name, follow)

    if name == "estimate_delivery_time" and _wants_both_periods(folded):
        seen = [args for called, args, _result in done if called == "estimate_delivery_time"]
        if not any(args.get("hour_from") == 0 for args in seen) and name in available:
            distance = _distance(folded)
            if distance is not None:
                return _call(name, _estimate_args(folded, distance, "night"))
        return _say(_compare_periods(done, persian))

    return _say(_render(name, arguments, result, done, user_text, persian))


def _render(
    name: str, arguments: dict, result: dict, done: list[tuple[str, dict, dict]], user_text: str, persian: bool
) -> str:
    folded = user_text.translate(_DIGITS).lower()
    if name == "get_order_status" and _wants_cancel(folded):
        return _cancel_decision(arguments, result, persian)
    if name == "get_order_status":
        return _status_answer(arguments, result, persian)
    if name == "search_orders":
        return _search_answer(result, persian)
    if name == "list_my_orders":
        return _search_answer(result, persian)
    if name in ("estimate_delivery_time", "estimate_price_range"):
        return _stats_answer(name, result, persian)
    if name == "search_knowledge_base":
        return _knowledge_answer(result, user_text, persian)
    if name in ("create_support_ticket", "request_order_cancellation"):
        return _write_answer(name, result, persian)
    if result.get("error") == "invalid_arguments":
        return (
            "این ورودی را نمی‌توانم به تخمین تبدیل کنم."
            if persian
            else "I will not estimate a time from that input."
        )
    return OUT_OF_SCOPE_FA if persian else OUT_OF_SCOPE_EN


def greeting(available: set[str], persian: bool) -> str:
    chosen = [item for item in _SERVICES if not available or item[0] in available] or list(_SERVICES)
    lines = [item[1] if persian else item[2] for item in chosen]
    names = {item[0] for item in chosen}
    example = next((item for item in _EXAMPLES if not names or item[0] in names), None)
    if persian:
        text = "سلام. من دستیار ارسال الوپیک هستم و این کارها را می‌توانم برایتان انجام دهم:\n- " + "\n- ".join(lines)
    else:
        text = "Hello. I am the Alopeyk delivery assistant, and I can:\n- " + "\n- ".join(lines)
    if example is None:
        return text
    return text + "\n\n" + (example[1] if persian else example[2])


def _is_greeting(folded: str) -> bool:
    text = re.sub(r"^[\s!?.،,؟]+|[\s!?.،,؟]+$", "", folded.strip())
    return text in _GREETINGS


def _cancel_decision(arguments: dict, result: dict, persian: bool) -> str:
    order = result.get("order") or {}
    order_id = order.get("order_id") or arguments.get("order_id") or ""
    if not result.get("ok") or not order:
        return _missing(order_id, persian)
    if order.get("picked_up_at") or order.get("status") not in ("searching_courier", "courier_assigned"):
        if persian:
            return f"سفارش {order_id} از تحویل‌گیری پیک گذشته است و دیگر لغو نمی‌شود. با پشتیبانی تماس بگیرید."
        return f"Order {order_id} is already past pickup, so it cannot be cancelled. Contact support."
    if persian:
        return f"سفارش {order_id} هنوز تحویل پیک نشده. برای لغو، در پیام بعدی «تأیید» را بفرستید."
    return f'Reply "Yes, confirm." to cancel order {order_id} before pickup.'


def _status_answer(arguments: dict, result: dict, persian: bool) -> str:
    order = result.get("order") or {}
    order_id = order.get("order_id") or arguments.get("order_id") or ""
    if not result.get("ok") or not order:
        return _missing(order_id, persian)
    status = order.get("status") or ""
    notes = order.get("notes") or ""
    if "ignore all" in notes.lower() or "system notice" in notes.lower():
        notes = ""
    gloss = _STATUS_FA.get(status, status)
    if persian:
        text = f"وضعیت سفارش {order_id} «{gloss}» است ({status}). این را از پایگاه سفارش‌ها خواندم."
    else:
        text = f"Order {order_id} is {status}. This comes from the orders database."
    if notes:
        text += " " + notes[:180]
    return text


def _missing(order_id: str, persian: bool) -> str:
    if persian:
        return f"سفارش {order_id} را در این حساب پیدا نکردم.".strip()
    if order_id:
        return f"I could not find order {order_id} on this account."
    return "I could not find that order on this account."


def _search_answer(result: dict, persian: bool) -> str:
    return format_own_orders(result.get("orders") or [], persian)


def _stats_answer(name: str, result: dict, persian: bool) -> str:
    return estimate_reply(name, result, persian)


def _compare_periods(done: list[tuple[str, dict, dict]], persian: bool) -> str:
    evening = night = None
    for _name, args, result in done:
        if result.get("insufficient_data") or not result.get("in_transit_minutes"):
            continue
        if args.get("hour_from") == 17:
            evening = result
        elif args.get("hour_from") == 0:
            night = result
    if not evening or not night:
        return _stats_answer("estimate_delivery_time", evening or night or {"insufficient_data": True, "ok": True}, persian)
    eve = evening["in_transit_minutes"]["median"]
    late = night["in_transit_minutes"]["median"]
    relation = "slower" if eve > late else "faster"
    if persian:
        word = "کندتر" if eve > late else "سریع‌تر"
        return (
            f"عصر {word} است: میانهٔ زمان حمل عصر {eve} دقیقه و شب {late} دقیقه. "
            f"تعداد نمونه {evening['sample_size']} و {night['sample_size']}. این یک تضمین نیست."
        )
    return (
        f"The evening rush is {relation}: in-transit median {eve} minutes versus {late} minutes at night. "
        f"Sample size {evening['sample_size']} and {night['sample_size']}. This is not a guarantee."
    )


def _knowledge_answer(result: dict, user_text: str, persian: bool) -> str:
    passages = result.get("results") or []
    for passage in passages:
        if _relevant(user_text, passage.get("text") or ""):
            snippet = _snippet(passage.get("text") or "", user_text, persian)
            source = passage.get("source_url") or ""
            if persian:
                return f"{snippet}\nمنبع: {source}"
            return f"{snippet}\nSource: {source}"
    if persian:
        return "این را در راهنما پیدا نکردم."
    return "I could not find that in the help pages."


def _write_answer(name: str, result: dict, persian: bool) -> str:
    if result.get("status") == "created":
        ticket = result.get("ticket_id")
        order_id = result.get("order_id") or ""
        if persian:
            return f"تیکت {ticket} برای سفارش {order_id} ثبت شد."
        return f"Support ticket {ticket} is open for order {order_id}."
    if result.get("status") == "cancelled":
        order_id = result.get("order_id") or ""
        if persian:
            return f"سفارش {order_id} لغو شد."
        return f"Order {order_id} is cancelled."
    if result.get("error") == "cancellation_not_allowed":
        if persian:
            return "این سفارش بعد از تحویل‌گیری پیک لغو نمی‌شود."
        return "This order cannot be cancelled after pickup."
    if persian:
        return "این کار انجام نشد."
    return "I could not complete that action."


def _ask_ticket(order_id: str, persian: bool) -> tuple[str, list[ToolCall]]:
    if persian:
        return _say(f"برای ثبت تیکت سفارش {order_id}، در پیام بعدی «تأیید» را بفرستید.")
    return _say(f'Reply "Yes, confirm." to open the support ticket for order {order_id}.')


def _start_write(previous: str, available: set[str], persian: bool) -> tuple[str, list[ToolCall]]:
    folded = previous.translate(_DIGITS).lower()
    order_id = _order_id(previous)
    if not order_id:
        return _say(CLARIFY_FA if persian else CLARIFY_EN)
    if _wants_ticket(folded):
        return _maybe_call(
            "create_support_ticket",
            {"order_id": order_id, "message": previous[:500]},
            available,
            persian,
        )
    if _wants_cancel(folded):
        return _maybe_call(
            "request_order_cancellation",
            {"order_id": order_id, "reason": "Customer confirmed in chat."},
            available,
            persian,
        )
    return _say(CLARIFY_FA if persian else "What should I confirm?")


def _maybe_call(name: str, arguments: dict, available: set[str], persian: bool) -> tuple[str, list[ToolCall]]:
    if name not in available:
        if persian:
            return _say("این کار برای این حساب در دسترس نیست.")
        return _say("That action is not available for this account.")
    return _call(name, arguments)


def _call(name: str, arguments: dict) -> tuple[str, list[ToolCall]]:
    call = ToolCall(
        id="call-" + uuid.uuid4().hex[:8],
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )
    return "", [call]


def _say(text: str) -> tuple[str, list[ToolCall]]:
    return text, []


def _estimate_args(folded: str, distance: float, window: str | None) -> dict:
    arguments: dict = {"distance_km": distance}
    vehicle = _vehicle(folded)
    city = _city(folded)
    if vehicle:
        arguments["vehicle_type"] = vehicle
    if city:
        arguments["city"] = city
    if window == "evening":
        arguments["hour_from"] = 17
        arguments["hour_to"] = 20
    elif window == "night":
        arguments["hour_from"] = 0
        arguments["hour_to"] = 5
    return arguments


def _last_user(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _previous_user(messages: list[ChatMessage]) -> str:
    seen = False
    for message in reversed(messages):
        if message.role != "user":
            continue
        if not seen:
            seen = True
            continue
        return message.content
    return ""


def _asked_to_confirm(messages: list[ChatMessage]) -> bool:
    latest = None
    for index, message in enumerate(messages):
        if message.role == "user":
            latest = index
    if latest is None:
        return False
    for message in reversed(messages[:latest]):
        if message.role == "assistant" and message.content:
            return "Yes, confirm." in message.content or "«تأیید»" in message.content
    return False


def _observations(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if message.role == "user":
            return False
        if message.role == "tool":
            return True
    return False


def _turn_results(messages: list[ChatMessage]) -> list[tuple[str, dict, dict]]:
    last_user = -1
    for index, message in enumerate(messages):
        if message.role == "user":
            last_user = index
    pending: dict[str, ToolCall] = {}
    rows: list[tuple[str, dict, dict]] = []
    for message in messages[last_user + 1 :]:
        for call in message.tool_calls:
            pending[call.id] = call
        if message.role != "tool":
            continue
        call = pending.get(message.tool_call_id or "")
        if call is None:
            continue
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            arguments = {}
        try:
            result = json.loads(message.content)
        except json.JSONDecodeError:
            result = {"ok": False, "error": "malformed_tool_call"}
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(result, dict):
            result = {"ok": False}
        rows.append((call.name, arguments, result))
    return rows


def _history_order_ids(messages: list[ChatMessage]) -> list[str]:
    latest = None
    for index, message in enumerate(messages):
        if message.role == "user":
            latest = index
    end = latest if latest is not None else len(messages)
    found: list[str] = []
    for message in messages[:end]:
        if message.role == "system":
            continue
        found.extend(_order_id(message.content, all_ids=True))
    return found


def _order_id(text: str, *, all_ids: bool = False) -> str | list[str] | None:
    matches = [match.group(0).upper() for match in _ORDER.finditer(text.translate(_DIGITS))]
    if all_ids:
        return matches
    return matches[0] if matches else None


def _distance(folded: str) -> float | None:
    match = _DISTANCE.search(folded)
    if not match:
        return None
    return float(match.group(1))


def _vehicle(folded: str) -> str | None:
    if "motorcycle" in folded or "موتور" in folded:
        return "motorcycle"
    if re.search(r"\bcar\b", folded) or "ماشین" in folded or "خودرو" in folded:
        return "car"
    if re.search(r"\bvan\b", folded) or "وانت" in folded:
        return "van"
    return None


def _city(folded: str) -> str | None:
    if "tehran" in folded or "تهران" in folded:
        return "Tehran"
    if "karaj" in folded or "کرج" in folded:
        return "Karaj"
    return None


def _directory_status(folded: str) -> str | None:
    if "in transit" in folded or "in-transit" in folded or "در حال حمل" in folded:
        return "in_transit"
    if "cancelled" in folded or "لغو" in folded:
        return "cancelled_by_customer"
    return None


def _wants_both_periods(folded: str) -> bool:
    return two_periods(folded)


def _wants_eta(folded: str) -> bool:
    return any(word in folded for word in ("how long", "طول", "تحویل", "delivery", "usually", "take"))


def _wants_price(folded: str) -> bool:
    return ("price" in folded or "قیمت" in folded) and "how long" not in folded and "طول" not in folded


def _wants_ticket(folded: str) -> bool:
    return "ticket" in folded or "تیکت" in folded


def _wants_cancel(folded: str) -> bool:
    return bool(re.search(r"\bcancel\b", folded)) or "لغو" in folded


def _wants_status(folded: str) -> bool:
    return any(word in folded for word in ("where", "status", "وضعیت", "کجاست", "چک", "بررسی", "summar", "note", "یادداشت"))


def _asks_phone(folded: str) -> bool:
    if "phone" in folded or "تلفن" in folded:
        return True
    return "شماره" in folded and "سفارش" not in folded


def _asks_directory(folded: str) -> bool:
    return any(
        phrase in folded
        for phrase in ("which orders", "all orders", "list all", "search all", "orders in ", "همه سفارش")
    )


def _refers_to_an_order(folded: str) -> bool:
    return any(phrase in folded for phrase in ("that order", "this order", "the order", "اون سفارش", "آن سفارش", "همان سفارش"))


def _wants_own_orders(folded: str) -> bool:
    return own_orders_requested(folded)


def _wants_knowledge(folded: str) -> bool:
    return any(
        phrase in folded
        for phrase in (
            "can i",
            "package",
            "fragile",
            "insurance",
            "payout",
            "چطور",
            "چگونه",
            "بسته",
            "شکستنی",
            "بیمه",
            "راهنما",
        )
    )


def _is_confirmation(folded: str) -> bool:
    cleaned = re.sub(r"[^\w\s\u0600-\u06FF]", "", folded, flags=re.UNICODE)
    cleaned = " ".join(cleaned.split())
    return cleaned in {"yes", "yes confirm", "confirm", "بله", "تایید", "تأیید", "بله تایید", "بله تأیید"}


def _keywords(question: str) -> set[str]:
    words = set(_WORD_EN.findall(question.lower())) | set(_WORD_FA.findall(question))
    return {word.lower() for word in words - _STOP}


def _relevant(question: str, passage: str) -> bool:
    return len(_hits(question, passage)) >= 2


def _hits(question: str, passage: str) -> list[str]:
    haystack = passage.translate(_DIGITS).lower()
    return [word for word in _keywords(question) if word in haystack]


def _snippet(text: str, question: str, persian: bool) -> str:
    parts = [part.strip() for part in re.split(r"(?m)^## ", text) if part.strip()]
    matching = [part for part in parts if _hits(question, part)] or parts or [text]
    if persian:
        preferred = [part for part in matching if _PERSIAN.search(part)]
    else:
        preferred = [part for part in matching if not _PERSIAN.search(part)]
    return (preferred or matching)[0][:400].strip()
