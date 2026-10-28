
# Use Python 3.12 to match the project environment.
FROM python:3.12-slim

WORKDIR /app

# Copy the frontend deployment files.
COPY . .

# Install dependencies and check compatibility.
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip check

EXPOSE 8501

# Start Streamlit with its default XSRF protection.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
