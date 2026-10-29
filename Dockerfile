FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# bundled seed data (box scores + trained models) is copied onto the persistent volume on first boot
RUN mkdir -p /seed && cp -r data/. /seed/ 2>/dev/null || true
ENV DATA_DIR=/data AUTO_REFRESH=1 PORT=8080
CMD ["sh", "-c", "[ -f /data/nba.db ] || cp -r /seed/. /data/; uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
