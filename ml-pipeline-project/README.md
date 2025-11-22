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

MIT License

## 🤝 Вклад

Pull requests приветствуются! Для серьезных изменений сначала откройте issue.
