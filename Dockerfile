# Dockerfile
FROM python:3.11-slim

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]