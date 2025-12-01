FROM python:3.11-slim

RUN pip install --no-cache-dir numpy scipy scikit-learn

WORKDIR /app
COPY docker/sandbox_runner.py .

CMD ["python", "sandbox_runner.py"]