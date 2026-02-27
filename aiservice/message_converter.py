from typing import Iterable, Optional

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage


def _get_value(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def to_langchain_messages(
    raw_messages: Optional[Iterable],
    system_prompt: Optional[str] = None,
    fallback_message: Optional[str] = None,
):
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    for item in raw_messages or []:
        role = (_get_value(item, "role") or "").lower()
        content = _get_value(item, "content")
        if content is None:
            continue
        if role in ("system", "sys"):
            messages.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    if fallback_message and fallback_message.strip():
        messages.append(HumanMessage(content=fallback_message))

    return messages
