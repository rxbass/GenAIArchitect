import re


def validate_output(
    answer: str,
) -> tuple[bool, str]:

    if not answer:
        return (
            False,
            "No answer was generated.",
        )

    if len(answer) > 10000:
        return (
            False,
            "Generated answer is too long.",
        )

    dangerous_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"developer message",
        r"internal instructions",
    ]

    lowered = answer.lower()

    for pattern in dangerous_patterns:

        if re.search(
            pattern,
            lowered,
        ):

            return (
                False,
                "The generated response failed "
                "the output safety check.",
            )

    return True, answer