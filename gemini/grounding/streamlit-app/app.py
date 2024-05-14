import streamlit as st
from vertexai.preview import rag
import vertexai

from vertexai.preview.generative_models import (
    GenerativeModel,
    Tool,
)

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "us-central1"
model_name = st.secrets["MODEL_NAME"]
rag_corpus_id = st.secrets["RAG_CORPUS_ID"]

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Initialize the RAG model
rag_model = GenerativeModel(
    model_name=model_name,
    tools=[
        Tool.from_retrieval(
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
        response = chat.send_message(prompt)
        full_response = response.text

        # Output the model response
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus_id,
                )
            ],
            text=prompt,
            similarity_top_k=1,
        )

    backend_details = ""

    for context in response.contexts.contexts:
        backend_details += f"""
- **Source URI:** {context.source_uri}
- **Distance:** `{context.distance}`
- **Text:** {context.text}...

---
"""

    with st.expander("**Retrieved Contexts:**"):
        st.markdown(backend_details)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "backend_details": backend_details,
        }
    )
