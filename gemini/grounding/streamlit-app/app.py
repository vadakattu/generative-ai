import streamlit as st
from google.cloud import rag
import vertexai

PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "us-central1"
rag_corpus_id = st.secrets["RAG_CORPUS_ID"]
model_name = "gemini-1.5-pro-preview-0514"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize the RAG model
rag_model = rag.GenerativeModel(
    model_name=model_name,
    tools=[
        rag.Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_corpora=[rag_corpus_id],
                    similarity_top_k=3,
                    vector_distance_threshold=0.5,
                ),
            ),
        )
    ],
    system_instruction=[
        "You are a helpful car manual chatbot. Answer the car owner's question about their car."
    ],
)

# Initialize chat session
chat = rag_model.start_chat()

# --------- STREAMLIT APP ---------------------------------------
st.title("🚗 Fix my car! ")
st.text("Questions about your vehicle? Ask me! Include the make, model, and year.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat history in UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Tell me the make + model of your car, then ask a question."
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # include previous messages as context
        for response in chat.send_message(prompt):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
