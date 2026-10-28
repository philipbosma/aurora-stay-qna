
import os

import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Configure this address for the deployment environment.
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://localhost:7860",
).rstrip("/")


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AuroraStay",
    page_icon="🏨",
    layout="wide",
)


# ---------------------------------------------------------
# Sidebar — Model parameters
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Parameters")

    st.caption(
        "Defaults match the evaluated configuration. "
        "You can adjust these settings for online inference."
    )

    k = st.slider(
        "k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Maximum number of relevant chunks used to answer the question.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values allow more variation in generated responses.",
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens in the generated answer.",
    )


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.title("🏨 AuroraStay")
st.write("Ask a question about AuroraStay hotel policies.")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True,
)


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder=(
        "e.g., What is the cancellation policy "
        "for an Advance Purchase booking?"
    ),
)


# ---------------------------------------------------------
# Submit
# ---------------------------------------------------------

if st.button("Submit", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:
        # Pick endpoint.
        if response_type == "Answer only":
            endpoint = "/v1/answer"
        else:
            endpoint = "/v1/answer_with_relevant_chunks"

        # Send the selected parameters to the backend.
        payload = {
            "query": query.strip(),
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with st.spinner("Thinking…"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=180,
                )

            if response.status_code == 200:
                result = response.json()

                # -- Answer only --------------------------------
                if response_type == "Answer only":
                    st.subheader("Answer")
                    st.write(result["answer"])

                # -- Answer + Relevant Chunks -------------------
                else:
                    st.subheader("Answer")
                    st.write(result["answer"])

                    chunks = result["relevant_chunks"]

                    st.subheader("Relevant Chunks")
                    st.caption(f"Returned chunks: {len(chunks)}")

                    for i, chunk in enumerate(chunks, start=1):
                        with st.expander(f"Chunk {i}"):
                            st.write(chunk)

            else:
                st.error(
                    "Unable to get a response from the RAG API. "
                    f"Status code: {response.status_code}"
                )

                # Display validation details when available.
                try:
                    error_details = response.json()
                except ValueError:
                    error_details = {}

                if isinstance(error_details, dict):
                    error_message = error_details.get("error")

                    if error_message:
                        st.error(error_message)

        except requests.exceptions.Timeout:
            st.error(
                "The RAG API took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the RAG API. "
                "Please make sure the backend service is running."
            )
