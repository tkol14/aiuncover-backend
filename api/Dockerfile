# Dockerfile (root, коли код у ./api)
FROM python:3.11-bookworm

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# залежності з підпапки
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# код із підпапки
COPY api/app ./app

ENV PORT=8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
