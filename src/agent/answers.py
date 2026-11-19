"""Sentences that restate a tool result.

The live turn and the stub both use this module. The agent loop does not.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from src.llm.types import ChatMessage, ToolCall

STATUS_FA = {
    "searching_courier": "در جستجوی پیک",
    "courier_assigned": "پیک تعیین شده",
    "in_transit": "در حال حمل",
    "delivered": "تحویل‌شده",
    "cancelled_by_customer": "لغو شده توسط مشتری",
    "cancelled_no_courier": "لغو شده چون پیکی نبود",
}

_PERSIAN = re.compile(r"[\u0600-\u06FF]")
_ORDER_ID = re.compile(r"A-\d+", re.IGNORECASE)
_ESTIMATES = {"estimate_delivery_time", "estimate_price_range"}
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
Execute = Callable[[str, dict[str, Any]], dict[str, Any]]


def format_own_orders(rows: list, persian: bool) -> str:
    shown: list[str] = []
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        order_id = str(row.get("order_id") or "")
        if not order_id:
            continue
        status = str(row.get("status") or "")
        label = STATUS_FA.get(status, status) if persian else status
        shown.append(f"{order_id} ({label})" if label else order_id)
    if not shown:
        if persian:
            return "سفارشی برای این حساب در پایگاه سفارش‌ها پیدا نکردم."
        return "I could not find any orders on this account in the orders database."
    body = "، ".join(shown) if persian else ", ".join(shown)
    if persian:
        return f"این سفارش‌ها را از پایگاه سفارش‌ها خواندم: {body}."
    return f"These orders come from the orders database: {body}."


def estimate_reply(name: str, result: dict, persian: bool) -> str:
    if not result.get("ok") or result.get("insufficient_data") or result.get("error"):
        if persian:
            return "داده کافی برای این تخمین وجود ندارد، پس عددی اعلام نمی‌کنم. این یک تضمین نیست."
        return "There is not enough data for that estimate, so I will not give a number. This is not a guarantee."
    sample = result.get("sample_size")
    if name == "estimate_price_range":
        price = result.get("price_toman") or {}
        if persian:
            return (
                f"میانهٔ قیمت {price.get('median')} تومان است و بازهٔ معمول تا {price.get('p80')} تومان. "
                f"تعداد نمونه {sample} سفارش تحویل‌شده از اسنپ‌شات. این یک تضمین نیست."
            )
        return (
            f"The median price is {price.get('median')} toman (typical range up to {price.get('p80')}). "
            f"Sample size {sample} delivered orders from the snapshot. This is not a guarantee."
        )
    transit = result.get("in_transit_minutes") or {}
    total = result.get("total_minutes") or {}
    if persian:
        return (
            f"میانهٔ زمان حمل {transit.get('median')} دقیقه است (بازهٔ معمول تا {transit.get('p80')} دقیقه). "
            f"زمان کل از ثبت سفارش میانهٔ {total.get('median')} دقیقه است. "
            f"تعداد نمونه {sample} سفارش تحویل‌شده از اسنپ‌شات. این یک تضمین نیست."
        )
    return (
        f"Once the courier has picked up the parcel, the in-transit median is {transit.get('median')} minutes "
        f"(typical range up to {transit.get('p80')} minutes). "
        f"Total time since the order was created has median {total.get('median')} minutes. "
        f"Sample size {sample} delivered orders from the snapshot. This is not a guarantee."
    )


def own_orders_requested(text: str) -> bool:
    folded = text.translate(_DIGITS).lower()
    if _wants_knowledge(folded) or _wants_cancel(folded) or _wants_ticket(folded):
        return False
    if any(
        phrase in folded
        for phrase in ("my orders", "my order", "track my", "سفارش‌های من", "سفارشهای من", "سفارشم")
    ):
        return True
    if "پیگیری" in folded and "سفارش" in folded:
        return True
    return "سفارش" in folded and any(
        word in folded for word in ("نمیدانم", "نمی‌دانم", "نمیدونم", "نمی دونم", "ندارم")
    )


def two_periods(text: str) -> bool:
    folded = text.translate(_DIGITS).lower()
    evening = "evening" in folded or "17:00" in folded or "عصر" in folded
    night = "night" in folded or "00:00" in folded or "شب" in folded
    return evening and night


def read_own_orders(
    messages: list[ChatMessage], tools: list[dict], execute: Execute
) -> tuple[str, list[ChatMessage], list[str]] | None:
    """Read this customer's orders before the model speaks.

    A small local model otherwise asks for an order id, including after the
    user says they do not know it. The lookup is already allowed for the role.
    """
    if not any(tool.get("name") == "list_my_orders" for tool in tools):
        return None
    user = _latest_user(messages)
    if not user or _ORDER_ID.search(user) or not own_orders_requested(user):
        return None
    transcript = list(messages)
    call = ToolCall(id="call-orders", name="list_my_orders", arguments='{"limit": 5}')
    transcript.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
    observation = _run_call(call, execute)
    transcript.append(
        ChatMessage(
            role="tool",
            content=json.dumps(observation, ensure_ascii=False),
            tool_call_id=call.id,
            name=call.name,
        )
    )
    rows = observation.get("orders") if isinstance(observation, dict) else None
    reply = format_own_orders(rows if isinstance(rows, list) else [], _persian(transcript))
    transcript.append(ChatMessage(role="assistant", content=reply))
    return reply, transcript[len(messages) :], [call.name]


def reply_after_tools(transcript: list[ChatMessage]) -> str | None:
    """Answer from the estimate tool instead of another model call.

    One sample decides a single estimate. Two time windows decide the comparison.
    """
    found = _estimate_details(transcript)
    if two_periods(_latest_user(transcript)):
        return _compare_from_results(found, _persian(transcript))
    if len(found) != 1:
        return None
    name, _arguments, result = found[0]
    if not result.get("ok"):
        return None
    if result.get("insufficient_data") or _estimate_figure(name, result) is not None:
        return estimate_reply(name, result, _persian(transcript))
    return None


def finish_reply(transcript: list[ChatMessage], content: str) -> str:
    """Use the tool's median when the model answers an estimate without it.

    A local model can call the tool and then say the data is missing. The
    snapshot figure is the answer the user should see.
    """
    found: list[tuple[str, dict[str, Any], float]] = []
    for name, _arguments, result in _estimate_details(transcript):
        if not result.get("ok") or result.get("insufficient_data"):
            continue
        figure = _estimate_figure(name, result)
        if figure is not None:
            found.append((name, result, figure))
    if len(found) != 1:
        return content
    name, result, figure = found[0]
    if _mentions_number(content, figure):
        return content
    return estimate_reply(name, result, _persian(transcript))


def _estimate_details(transcript: list[ChatMessage]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    last_user = -1
    for index, message in enumerate(transcript):
        if message.role == "user":
            last_user = index
    pending: dict[str, ToolCall] = {}
    found: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for message in transcript[last_user + 1 :]:
        for call in message.tool_calls:
            pending[call.id] = call
        if message.role != "tool" or message.name not in _ESTIMATES:
            continue
        call = pending.get(message.tool_call_id or "")
        try:
            arguments = json.loads(call.arguments) if call is not None else {}
        except json.JSONDecodeError:
            arguments = {}
        try:
            result = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        if isinstance(result, dict):
            found.append((message.name, arguments, result))
    return found


def _compare_from_results(
    found: list[tuple[str, dict[str, Any], dict[str, Any]]], persian: bool
) -> str | None:
    evening = night = None
    for name, arguments, result in found:
        if name != "estimate_delivery_time":
            continue
        if arguments.get("hour_from") == 17:
            evening = result
        elif arguments.get("hour_from") == 0:
            night = result
    if evening is None or night is None:
        return None
    return compare_periods(evening, night, persian)


def _estimate_figure(name: str, result: dict[str, Any]) -> float | None:
    payload = result.get("price_toman") if name == "estimate_price_range" else result.get("in_transit_minutes")
    if not isinstance(payload, dict):
        return None
    median = payload.get("median")
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        return None
    return float(median)


def _mentions_number(content: str, number: float) -> bool:
    text = f"{number:g}"
    folded = content.translate(_DIGITS)
    return re.search(rf"(?<!\d){re.escape(text)}(?!\d)", folded) is not None


def _run_call(call: ToolCall, execute: Execute) -> dict[str, Any]:
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return {"ok": False, "error": "malformed_tool_call", "message": "Tool arguments were not valid JSON."}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "malformed_tool_call", "message": "Tool arguments must be a JSON object."}
    try:
        result = execute(call.name, arguments)
    except Exception:
        return {"ok": False, "error": "tool_failed", "message": "The tool failed before it could return a result."}
    if not isinstance(result, dict):
        return {"ok": False, "error": "tool_failed", "message": "The tool returned an unexpected result."}
    return result


def _latest_user(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _persian(messages: list[ChatMessage]) -> bool:
    return bool(_PERSIAN.search(_latest_user(messages)))


def _wants_ticket(folded: str) -> bool:
    return "ticket" in folded or "تیکت" in folded


def _wants_cancel(folded: str) -> bool:
    return bool(re.search(r"\bcancel\b", folded)) or "لغو" in folded


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


_DISTANCE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:km|kilometers?|کیلومتر)")
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
CLARIFY_EN = "Which order id should I check?"
CLARIFY_FA = "شمارهٔ کدام سفارش را بررسی کنم؟"
Turn = tuple[str, list[ChatMessage], list[str]]


def compare_periods(evening: dict, night: dict, persian: bool) -> str:
    if (
        not evening
        or not night
        or evening.get("insufficient_data")
        or night.get("insufficient_data")
        or not evening.get("in_transit_minutes")
        or not night.get("in_transit_minutes")
    ):
        return estimate_reply(
            "estimate_delivery_time",
            evening or night or {"insufficient_data": True, "ok": True},
            persian,
        )
    eve = evening["in_transit_minutes"]["median"]
    late = night["in_transit_minutes"]["median"]
    if persian:
        word = "کندتر" if eve > late else "سریع‌تر"
        return (
            f"عصر {word} است: میانهٔ زمان حمل عصر {eve} دقیقه و شب {late} دقیقه. "
            f"تعداد نمونه {evening.get('sample_size')} و {night.get('sample_size')}. این یک تضمین نیست."
        )
    relation = "slower" if eve > late else "faster"
    return (
        f"The evening rush is {relation}: in-transit median {eve} minutes versus {late} minutes at night. "
        f"Sample size {evening.get('sample_size')} and {night.get('sample_size')}. This is not a guarantee."
    )


def knowledge_reply(result: dict, user_text: str, persian: bool) -> str:
    for passage in result.get("results") or []:
        if not isinstance(passage, dict):
            continue
        text = passage.get("text") or ""
        if _relevant(user_text, text):
            snippet = _snippet(text, user_text, persian)
            source = passage.get("source_url") or ""
            if persian:
                return f"{snippet}\nمنبع: {source}"
            return f"{snippet}\nSource: {source}"
    if persian:
        return "این را در راهنما پیدا نکردم."
    return "I could not find that in the help pages."


def prepare_turn(messages: list[ChatMessage], tools: list[dict], execute: Execute) -> Turn | None:
    """Answer a turn from tools before a live model can skip them.

    The stub already follows these branches. A smaller or hosted model does not,
    so the same lookup runs here and the reply is the tool sentence.
    """
    names = {str(tool.get("name")) for tool in tools}
    user = _latest_user(messages)
    if not user:
        return None
    folded = user.translate(_DIGITS).lower()
    persian = bool(_PERSIAN.search(user))

    if _is_confirmation(folded):
        return _confirmed_write(messages, names, execute, persian)

    if "drop table" in folded or "distance_km" in folded:
        if "estimate_delivery_time" not in names:
            return None
        quoted = re.search(r'distance_km\s*=\s*"([^"]*)"', user, re.IGNORECASE)
        raw = quoted.group(1) if quoted else "invalid"
        return _one_call(
            messages,
            execute,
            "estimate_delivery_time",
            {"distance_km": raw, "vehicle_type": "motorcycle"},
            lambda result: estimate_reply("estimate_delivery_time", result, persian),
        )

    if _asks_phone(folded):
        text = "شمارهٔ تلفن مشتری را نمی‌توانم بگویم." if persian else "I can't share customer phone numbers."
        return _say(messages, text)

    order_id = _one_order_id(user)
    if order_id and _wants_ticket(folded) and "create_support_ticket" in names:
        ask = (
            f"برای ثبت تیکت سفارش {order_id}، در پیام بعدی «تأیید» را بفرستید."
            if persian
            else f'Reply "Yes, confirm." to open the support ticket for order {order_id}.'
        )
        return _say(messages, ask)
    if order_id and _wants_cancel(folded) and "get_order_status" in names:
        return _one_call(
            messages,
            execute,
            "get_order_status",
            {"order_id": order_id},
            lambda result: _cancel_reply(order_id, result, persian),
        )
    if order_id and (_wants_status(folded) or not _wants_knowledge(folded)) and "get_order_status" in names:
        return _one_call(
            messages,
            execute,
            "get_order_status",
            {"order_id": order_id},
            lambda result: _status_reply(order_id, result, persian),
        )

    if not order_id and _refers_to_an_order(folded):
        prior = _history_order_ids(messages)
        if not prior or "get_order_status" not in names:
            return _say(messages, CLARIFY_FA if persian else CLARIFY_EN)
        chosen = prior[0] if "first" in folded or "اول" in user else prior[-1]
        return _one_call(
            messages,
            execute,
            "get_order_status",
            {"order_id": chosen},
            lambda result: _status_reply(chosen, result, persian),
        )

    if _asks_directory(folded):
        if "search_orders" not in names:
            text = (
                "این حساب فقط سفارش‌های خودش را می‌بیند. شمارهٔ سفارش را بفرستید."
                if persian
                else "This account can only see its own orders. Send an order id."
            )
            return _say(messages, text)
        arguments: dict[str, Any] = {"limit": 20}
        city = _city(folded)
        status = _directory_status(folded)
        if city:
            arguments["city"] = city
        if status:
            arguments["status"] = status
        if len(arguments) == 1:
            arguments["status"] = "in_transit"
        return _one_call(
            messages,
            execute,
            "search_orders",
            arguments,
            lambda result: format_own_orders(result.get("orders") or [], persian),
        )

    distance = _distance(folded)
    if distance is not None and _wants_price(folded) and "estimate_price_range" in names:
        return _one_call(
            messages,
            execute,
            "estimate_price_range",
            _estimate_args(folded, distance, None),
            lambda result: estimate_reply("estimate_price_range", result, persian),
        )
    if distance is not None and _wants_eta(folded) and "estimate_delivery_time" in names:
        return _estimate_turn(messages, execute, folded, distance, persian)

    if _wants_knowledge(folded) and "search_knowledge_base" in names:
        return _one_call(
            messages,
            execute,
            "search_knowledge_base",
            {"query": user, "limit": 3},
            lambda result: knowledge_reply(result, user, persian),
        )
    return None


def _estimate_turn(
    messages: list[ChatMessage], execute: Execute, folded: str, distance: float, persian: bool
) -> Turn:
    if not two_periods(folded):
        return _one_call(
            messages,
            execute,
            "estimate_delivery_time",
            _estimate_args(folded, distance, None),
            lambda result: estimate_reply("estimate_delivery_time", result, persian),
        )
    evening = _estimate_args(folded, distance, "evening")
    night = _estimate_args(folded, distance, "night")
    first = _run_named(execute, "estimate_delivery_time", evening)
    second = _run_named(execute, "estimate_delivery_time", night)
    reply = compare_periods(first, second, persian)
    return _record(
        messages,
        [("estimate_delivery_time", evening, first), ("estimate_delivery_time", night, second)],
        reply,
    )


def _confirmed_write(
    messages: list[ChatMessage], names: set[str], execute: Execute, persian: bool
) -> Turn | None:
    if not _asked_to_confirm(messages):
        return _say(messages, CLARIFY_FA if persian else "What should I confirm?")
    previous = _previous_user(messages)
    folded = previous.translate(_DIGITS).lower()
    order_id = _one_order_id(previous)
    if not order_id:
        return _say(messages, CLARIFY_FA if persian else CLARIFY_EN)
    if _wants_ticket(folded) and "create_support_ticket" in names:
        return _write(messages, execute, "create_support_ticket", {"order_id": order_id, "message": previous[:500]}, persian)
    if _wants_cancel(folded) and "request_order_cancellation" in names:
        return _write(
            messages,
            execute,
            "request_order_cancellation",
            {"order_id": order_id, "reason": "Customer confirmed in chat."},
            persian,
        )
    return _say(messages, CLARIFY_FA if persian else "What should I confirm?")


def _write(
    messages: list[ChatMessage], execute: Execute, name: str, arguments: dict[str, Any], persian: bool
) -> Turn:
    pending = _run_named(execute, name, arguments)
    steps = [(name, arguments, pending)]
    if pending.get("status") == "confirmation_required" and pending.get("confirmation_id"):
        follow = dict(arguments)
        follow["confirmed"] = True
        follow["confirmation_id"] = pending["confirmation_id"]
        pending = _run_named(execute, name, follow)
        steps.append((name, follow, pending))
    return _record(messages, steps, _write_reply(name, pending, persian))


def _write_reply(name: str, result: dict, persian: bool) -> str:
    del name
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


def _status_reply(order_id: str, result: dict, persian: bool) -> str:
    order = result.get("order") or {}
    if not result.get("ok") or not isinstance(order, dict) or not order:
        if persian:
            return f"سفارش {order_id} را در این حساب پیدا نکردم.".strip()
        if order_id:
            return f"I could not find order {order_id} on this account."
        return "I could not find that order on this account."
    found = str(order.get("order_id") or order_id)
    status = str(order.get("status") or "")
    notes = str(order.get("notes") or "")
    if "ignore all" in notes.lower() or "system notice" in notes.lower():
        notes = ""
    gloss = STATUS_FA.get(status, status)
    if persian:
        text = f"وضعیت سفارش {found} «{gloss}» است ({status}). این را از پایگاه سفارش‌ها خواندم."
    else:
        text = f"Order {found} is {status}. This comes from the orders database."
    if notes:
        text += " " + notes[:180]
    return text


def _cancel_reply(order_id: str, result: dict, persian: bool) -> str:
    order = result.get("order") if isinstance(result.get("order"), dict) else {}
    found = str(order.get("order_id") or order_id)
    if not result.get("ok") or not order:
        return _status_reply(order_id, result, persian)
    if order.get("picked_up_at") or order.get("status") not in ("searching_courier", "courier_assigned"):
        if persian:
            return f"سفارش {found} از تحویل‌گیری پیک گذشته است و دیگر لغو نمی‌شود. با پشتیبانی تماس بگیرید."
        return f"Order {found} is already past pickup, so it cannot be cancelled. Contact support."
    if persian:
        return f"سفارش {found} هنوز تحویل پیک نشده. برای لغو، در پیام بعدی «تأیید» را بفرستید."
    return f'Reply "Yes, confirm." to cancel order {found} before pickup.'


def _one_call(
    messages: list[ChatMessage],
    execute: Execute,
    name: str,
    arguments: dict[str, Any],
    render: Callable[[dict[str, Any]], str],
) -> Turn:
    result = _run_named(execute, name, arguments)
    return _record(messages, [(name, arguments, result)], render(result))


def _record(
    messages: list[ChatMessage],
    steps: list[tuple[str, dict[str, Any], dict[str, Any]]],
    reply: str,
) -> Turn:
    transcript = list(messages)
    names: list[str] = []
    for index, (name, arguments, result) in enumerate(steps, start=1):
        call = ToolCall(
            id=f"call-prep-{index}",
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
        names.append(name)
        transcript.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
        transcript.append(
            ChatMessage(
                role="tool",
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=call.id,
                name=name,
            )
        )
    transcript.append(ChatMessage(role="assistant", content=reply))
    return reply, transcript[len(messages) :], names


def _say(messages: list[ChatMessage], text: str) -> Turn:
    transcript = [*messages, ChatMessage(role="assistant", content=text)]
    return text, transcript[len(messages) :], []


def _run_named(execute: Execute, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    call = ToolCall(id="call-run", name=name, arguments=json.dumps(arguments, ensure_ascii=False))
    return _run_call(call, execute)


def _estimate_args(folded: str, distance: float, window: str | None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"distance_km": distance}
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
        found.extend(_all_order_ids(message.content))
    return found


def _one_order_id(text: str) -> str | None:
    matches = _all_order_ids(text)
    return matches[0] if matches else None


def _all_order_ids(text: str) -> list[str]:
    return [match.group(0).upper() for match in _ORDER_ID.finditer(text.translate(_DIGITS))]


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


def _wants_eta(folded: str) -> bool:
    return any(word in folded for word in ("how long", "طول", "تحویل", "delivery", "usually", "take"))


def _wants_price(folded: str) -> bool:
    return ("price" in folded or "قیمت" in folded) and "how long" not in folded and "طول" not in folded


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
    return any(
        phrase in folded for phrase in ("that order", "this order", "the order", "اون سفارش", "آن سفارش", "همان سفارش")
    )


def _is_confirmation(folded: str) -> bool:
    cleaned = re.sub(r"[^\w\s\u0600-\u06FF]", "", folded, flags=re.UNICODE)
    cleaned = " ".join(cleaned.split())
    return cleaned in {"yes", "yes confirm", "confirm", "بله", "تایید", "تأیید", "بله تایید", "بله تأیید"}


def _relevant(question: str, passage: str) -> bool:
    return len(_hits(question, passage)) >= 2


def _hits(question: str, passage: str) -> list[str]:
    haystack = passage.translate(_DIGITS).lower()
    words = set(_WORD_EN.findall(question.lower())) | set(_WORD_FA.findall(question))
    keywords = {word.lower() for word in words - _STOP}
    return [word for word in keywords if word in haystack]


def _snippet(text: str, question: str, persian: bool) -> str:
    parts = [part.strip() for part in re.split(r"(?m)^## ", text) if part.strip()]
    matching = [part for part in parts if _hits(question, part)] or parts or [text]
    if persian:
        preferred = [part for part in matching if _PERSIAN.search(part)]
    else:
        preferred = [part for part in matching if not _PERSIAN.search(part)]
    return (preferred or matching)[0][:400].strip()
