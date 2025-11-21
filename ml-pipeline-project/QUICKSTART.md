# 🚀 БЫСТРЫЙ СТАРТ

Краткая инструкция для запуска ML Pipeline за 5 минут.

## Шаг 1: Установка (2 минуты)

```bash
# Клонирование
git clone https://github.com/your-username/ml-pipeline-project.git
cd ml-pipeline-project

# Установка зависимостей
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Шаг 2: Запуск полного pipeline (3 минуты)

```bash
bash scripts/run_pipeline.sh
```

Этот скрипт автоматически:
1. ✅ Проверит данные с Deepchecks
2. ✅ Проанализирует дрейф с EvidentlyAI
3. ✅ Обучит модель с MLflow

## Шаг 3: Просмотр результатов

### MLflow UI
```bash
mlflow ui
```
Откройте: http://localhost:5000

### Отчеты
- `reports/deepchecks_*.html` - валидация данных
- `reports/evidently_*.html` - мониторинг дрейфа

## Запуск отдельных компонентов

```bash
# Только валидация данных
python src/deepchecks_validation.py

# Только мониторинг дрейфа
python src/evidently_monitoring.py

# Только обучение модели
python src/train.py --model-type rf --n-estimators 100 --max-depth 10
```

## Docker (опционально)

```bash
# Сборка
docker build -t ml-pipeline .

# Запуск MLflow UI
docker run -p 5000:5000 ml-pipeline

# Обучение модели
docker run ml-pipeline python src/train.py
```

## CI/CD

### GitHub Actions
```bash
git add .
git commit -m "Add ML Pipeline"
git push origin main
```
Просмотр: https://github.com/your-username/ml-pipeline-project/actions

### GitLab CI
```bash
git remote add gitlab https://gitlab.com/your-username/ml-pipeline-project.git
git push gitlab main
```
Просмотр: https://gitlab.com/your-username/ml-pipeline-project/-/pipelines

## Тесты

```bash
pytest tests/ -v
```

## Полная документация

Смотрите:
- `README.md` - подробная документация
- `DEPLOYMENT_GUIDE.md` - пошаговая инструкция развертывания

## Помощь

Если что-то не работает:
1. Проверьте версию Python: `python --version` (нужна 3.8+)
2. Переустановите зависимости: `pip install -r requirements.txt --force-reinstall`
3. Посмотрите логи в терминале

## Результат

После выполнения всех шагов у вас будет:
- ✅ Обученная модель в `models/`
- ✅ Отчеты в `reports/`
- ✅ Эксперименты в `mlruns/`
- ✅ Метрики в MLflow UI

Время выполнения: ~5 минут
