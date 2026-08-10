# Input and display helpers for LegisBrief-NLP.

import re


MINIMUM_BILL_WORDS = 20
MAXIMUM_TEXT_CHARACTERS = 500_000


def normalize_input(value):
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_bill_input(
    title,
    bill_text,
    minimum_words=MINIMUM_BILL_WORDS
):
    normalized_title = normalize_input(title)
    normalized_bill_text = normalize_input(bill_text)

    errors = []
    warnings = []

    word_count = len(normalized_bill_text.split())
    character_count = len(normalized_bill_text)

    if not normalized_bill_text:
        errors.append("Bill text is required.")

    elif word_count < minimum_words:
        errors.append(
            f"Bill text must contain at least "
            f"{minimum_words} words."
        )

    if character_count > MAXIMUM_TEXT_CHARACTERS:
        errors.append(
            "Bill text is larger than the application "
            "character limit."
        )

    if normalized_bill_text and word_count > 1_500:
        warnings.append(
            "This is a long bill. The summarizer uses "
            "the configured maximum input-token limit."
        )

    return {
        "is_valid": not errors,
        "title": normalized_title,
        "bill_text": normalized_bill_text,
        "word_count": word_count,
        "character_count": character_count,
        "errors": errors,
        "warnings": warnings
    }
