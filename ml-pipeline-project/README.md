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

## Запустить только MLflow UI
```bash
sudo docker compose up -d mlflow-ui
```
Откройте: http://localhost:80

## Обучить модель
```bash
sudo docker compose run --rm train
```

## Запустить валидацию
```bash
sudo docker compose --profile validation run --rm deepchecks
```

## Запустить мониторинг
```bash
sudo docker compose --profile monitoring run --rm evidently
```

## Просмотр логов MLflow UI
```bash
sudo docker-compose logs -f mlflow-ui
```

## Остановить все
```bash
sudo docker-compose down
```
