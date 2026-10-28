
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN python -m pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch.
RUN python -m pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies and check package compatibility.
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip check

ENV HF_HOME=/app/huggingface

# Download the evaluated embedding model at build time.
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings( \
    model_name='HarishMaths/Hotel-Policy-Embedding', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True} \
)"

# Load the cached Hugging Face model at runtime.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

EXPOSE 7860

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
