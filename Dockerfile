FROM python:3.11-slim

# Встановимо системні залежності (за потреби Pillow/OpenCV тощо):
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Спочатку залежності (щоб кеш не ламався при зміні коду)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Потім код
COPY app ./app

# Railway надасть PORT у змінній середовища
ENV PORT=8000

# Запускаємо FastAPI через uvicorn
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
