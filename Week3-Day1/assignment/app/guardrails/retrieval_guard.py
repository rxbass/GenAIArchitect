def validate_retrieval(
    results: list[dict],
) -> tuple[bool, str]:

    if not isinstance(
        results,
        list,
    ):

        return (
            False,
            "Invalid graph result format.",
        )

    for result in results:

        if not isinstance(
            result,
            dict,
        ):

            return (
                False,
                "Invalid graph result.",
            )

    return True, ""