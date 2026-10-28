
import math
import os
from pathlib import Path

from flask import Flask, abort, jsonify, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from rag_utils import retrieve, rag, rewrite_policy_query


# ---------------------------------------------------------
# Configuration — evaluated defaults
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = Path(__file__).resolve().parent / "faiss_index"

RAG_SETTINGS = {
    "k": 5,
    "candidate_k": 20,
    "model_name": "gpt-4o-mini",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

# Load only the index created and trusted for this project.
vectorstore = FAISS.load_local(
    str(VECTORSTORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True,
)

if vectorstore.index.ntotal != 100:
    raise RuntimeError(
        "Expected the evaluated FAISS index with 100 chunks."
    )

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Validate requests; use evaluated settings as defaults
# ---------------------------------------------------------

def read_query():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        abort(
            400,
            description="The request must contain a JSON object.",
        )

    query = data.get("query")

    if not isinstance(query, str) or not query.strip():
        abort(
            400,
            description="Provide a non-empty query string.",
        )

    # Copy defaults so a request cannot modify shared settings.
    settings = RAG_SETTINGS.copy()

    k = data.get("k", settings["k"])

    if type(k) is not int or not 1 <= k <= 10:
        abort(
            400,
            description="k must be an integer from 1 to 10.",
        )

    temperature = data.get(
        "temperature",
        settings["temperature"],
    )

    if (
        type(temperature) not in (int, float)
        or not 0 <= temperature <= 1
        or not math.isfinite(temperature)
    ):
        abort(
            400,
            description="temperature must be a number from 0 to 1.",
        )

    max_tokens = data.get("max_tokens", settings["max_tokens"])

    if type(max_tokens) is not int or not 64 <= max_tokens <= 2048:
        abort(
            400,
            description="max_tokens must be an integer from 64 to 2048.",
        )

    # These settings remain fixed for every request.
    for name in ("candidate_k", "model_name", "top_p"):
        if name in data and data[name] != settings[name]:
            abort(
                400,
                description=f"{name} must remain {settings[name]}.",
            )

    settings.update(
        k=k,
        temperature=float(temperature),
        max_tokens=max_tokens,
    )

    return query, settings


@rag_api.errorhandler(400)
def bad_request(error):
    return jsonify({"error": error.description}), 400


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    query, settings = read_query()

    # Use the selected settings for query rewriting.
    model = ChatOpenAI(
        model=settings["model_name"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        max_tokens=settings["max_tokens"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    search_query = rewrite_policy_query(
        query=query,
        model=model,
    )

    chunks = retrieve(
        query=query,
        search_query=search_query,
        k=settings["k"],
        candidate_k=settings["candidate_k"],
        vectorstore=vectorstore,
    )

    return jsonify({
        "query": query,
        "k": settings["k"],
        "candidate_k": settings["candidate_k"],
        "relevant_chunks": chunks,
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    query, settings = read_query()

    answer, chunks = rag(
        query=query,
        vectorstore=vectorstore,
        **settings,
    )

    return jsonify({
        "query": query,
        **settings,
        "relevant_chunks": chunks,
        "answer": answer,
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    query, settings = read_query()

    generated_answer, _ = rag(
        query=query,
        vectorstore=vectorstore,
        **settings,
    )

    return jsonify({
        "answer": generated_answer,
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
    )
