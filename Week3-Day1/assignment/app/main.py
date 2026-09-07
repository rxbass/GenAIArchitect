from app.rag.kg_chat import KnowledgeGraphChat

from app.guardrails.input_guard import validate_user_input
from app.guardrails.output_guard import validate_output


def main():

    chatbot = KnowledgeGraphChat()

    print(
        "\nTamil Nadu Government Scheme Assistant"
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        question = input(
            "You: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print("Goodbye!")
            break

        # =====================================================
        # INPUT GUARDRAIL
        # =====================================================

        allowed, validated_question = validate_user_input(
            question
        )

        if not allowed:

            print(
                f"\nAssistant: {validated_question}\n"
            )

            continue

        # =====================================================
        # KNOWLEDGE GRAPH CHAT
        # =====================================================

        answer = chatbot.answer_question(
            validated_question
        )

        # =====================================================
        # OUTPUT GUARDRAIL
        # =====================================================

        output_allowed, validated_answer = validate_output(
            answer
        )

        if not output_allowed:

            print(
                "\nAssistant: "
                "I could not provide a safe response.\n"
            )

            continue

        print(
            "\nAssistant:",
            validated_answer,
        )

        print()


if __name__ == "__main__":
    main()