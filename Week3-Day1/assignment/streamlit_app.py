import streamlit as st

from app.rag.kg_chat import KnowledgeGraphChat
from app.guardrails.input_guard import validate_user_input
from app.guardrails.output_guard import validate_output


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tamil Nadu Government Scheme Assistant",
    page_icon="🏛️",
    layout="centered",
)


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("🏛️ Tamil Nadu Government Scheme Assistant")

st.caption(
    "Knowledge Graph powered assistant for Tamil Nadu Government schemes"
)


# ---------------------------------------------------------
# Initialize chatbot
# ---------------------------------------------------------

@st.cache_resource
def get_chatbot():

    return KnowledgeGraphChat()


chatbot = get_chatbot()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask about Tamil Nadu Government schemes..."
)


if question:

    # -----------------------------------------------------
    # Display user question
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # Input guardrail
    # -----------------------------------------------------

    allowed, validated_question = validate_user_input(
        question
    )

    if not allowed:

        answer = validated_question

    else:

        # -------------------------------------------------
        # Knowledge Graph pipeline
        # -------------------------------------------------

        with st.spinner("Searching the Knowledge Graph..."):

            answer = chatbot.answer_question(
                validated_question
            )

        # -------------------------------------------------
        # Output guardrail
        # -------------------------------------------------

        output_allowed, validated_answer = validate_output(
            answer
        )

        if not output_allowed:

            answer = (
                "I could not provide a safe response "
                "to this question."
            )

        else:

            answer = validated_answer


    # -----------------------------------------------------
    # Display assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)