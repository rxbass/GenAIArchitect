import re


def validate_user_input(
    question: str,
) -> tuple[bool, str]:

    question = question.strip()

    if not question:
        return False, "Please enter a question."

    if len(question) > 1000:
        return False, "Question is too long."

    suspicious_patterns = [
        r"\bignore previous instructions\b",
        r"\bignore all instructions\b",
        r"\bsystem prompt\b",
        r"\breveal your prompt\b",
        r"\bshow your instructions\b",
    ]

    lowered = question.lower()

    for pattern in suspicious_patterns:

        if re.search(
            pattern,
            lowered,
        ):

            return (
                False,
                "I can only answer questions about "
                "Tamil Nadu Government schemes."
            )

    return True, question