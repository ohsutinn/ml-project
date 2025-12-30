FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY bentofile.yaml ./bentofile.yaml

RUN chmod +x /app/app/entrypoint.sh
EXPOSE 3000
ENTRYPOINT ["/app/app/entrypoint.sh"]