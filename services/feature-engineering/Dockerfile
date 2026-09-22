FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY feature_engineering.py .
EXPOSE 8000
CMD ["uvicorn", "feature_engineering:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
