# Dockerfile (root, коли код лежить у ./api)
FROM python:3.11-bookworm

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# 1) залежності з підпапки api
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) код з підпапки api
COPY api/app ./app

# Railway підставить PORT, але ми явно вкажемо 8000
ENV PORT=8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
