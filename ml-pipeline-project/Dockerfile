# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Создаем директории для отчетов и MLflow
RUN mkdir -p /app/reports /app/mlruns

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Открываем порт для MLflow UI
EXPOSE 5000

# Метаданные
LABEL maintainer="Sergey <sergey@example.com>"
LABEL description="ML Pipeline with MLflow, Deepchecks, and Evidently AI"
LABEL version="1.0"

# Команда по умолчанию - запуск MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]

# Альтернативные команды для запуска:
# docker run ml-pipeline python src/train.py
# docker run ml-pipeline python src/deepchecks_validation.py
# docker run ml-pipeline python src/evidently_monitoring.py
