"""Single source of truth for the translation prompt.

Imported by prepare_data.py (training data) and evaluate.py (Stage 4), so the
model sees exactly the same instruction at train and eval time.
"""

PROMPT_TEMPLATE = "Translate the following English text to Telugu.\n\n{en}"


def build_messages(en: str, te: str | None = None) -> list[dict]:
    """Chat messages for one pair. Omit `te` to get an inference prompt."""
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(en=en)}]
    if te is not None:
        messages.append({"role": "assistant", "content": te})
    return messages
