# ML Pipeline: California Housing Prediction

Выполнил: Куриленко Сергей

Комплексный проект машинного обучения с полным CI/CD pipeline, включающий:
- MLflow для отслеживания экспериментов
- Deepchecks для валидации данных
- EvidentlyAI для мониторинга дрейфа данных
- GitHub Actions и GitLab CI для автоматизации
- Docker для контейнеризации


##  Быстрый старт

```bash
# Клонирование репозитория
git clone https://github.com/your-username/ml-pipeline-project.git
cd ml-pipeline-project

# Установка зависимостей
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Запуск полного pipeline
bash scripts/run_pipeline.sh

# Просмотр результатов MLflow
mlflow ui
# Откройте: http://localhost:5000
```

## Сборка Docker образа
```bash
docker build -t ml-pipeline:latest .
```

## Валидация данных:
```bash
docker run ml-pipeline:latest python src/deepchecks_validation.py
```

## Обучение модели:
```bash
docker run ml-pipeline:latest python src/train.py
```

MLflow UI:
```bash
docker run -p 80:5000 ml-pipeline:latest
```
Откройте: http://localhost:80

