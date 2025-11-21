# ML Pipeline: California Housing Prediction

Комплексный проект машинного обучения с полным CI/CD pipeline, включающий:
- ✅ MLflow для отслеживания экспериментов
- ✅ Deepchecks для валидации данных
- ✅ EvidentlyAI для мониторинга дрейфа данных
- ✅ GitHub Actions и GitLab CI для автоматизации
- ✅ Docker для контейнеризации

## 📋 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Структура проекта](#структура-проекта)
3. [Установка](#установка)
4. [Использование](#использование)
5. [CI/CD](#cicd)
6. [Docker](#docker)
7. [Детальная инструкция](#детальная-инструкция)

## 🚀 Быстрый старт

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

## 📁 Структура проекта

```
ml-pipeline-project/
├── src/                              # Исходный код
│   ├── train.py                     # Обучение модели с MLflow
│   ├── deepchecks_validation.py     # Проверка данных
│   └── evidently_monitoring.py      # Мониторинг дрейфа
├── tests/                            # Тесты
│   └── test_pipeline.py             # Unit-тесты
├── scripts/                          # Скрипты
│   └── run_pipeline.sh              # Полный запуск pipeline
├── data/                             # Данные (не в Git)
├── models/                           # Модели (Git LFS)
├── reports/                          # Отчеты (не в Git)
├── mlruns/                           # MLflow experiments
├── .github/workflows/                # GitHub Actions
│   └── ml-pipeline.yml              # CI/CD конфигурация
├── .gitlab-ci.yml                    # GitLab CI конфигурация
├── Dockerfile                        # Docker образ
├── MLproject                         # MLflow проект
├── conda.yaml                        # Conda окружение
├── requirements.txt                  # Python зависимости
├── .gitignore                        # Игнорируемые файлы
├── .gitattributes                    # Git LFS конфигурация
└── README.md                         # Документация
```

## 📦 Установка

### Требования

- Python 3.8+
- Git
- (Опционально) Docker
- (Опционально) Git LFS для больших файлов

### Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv

# Активация
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка пакетов
pip install -r requirements.txt
```

## 🎯 Использование

### 1. Проверка данных (Deepchecks)

```bash
cd src
python deepchecks_validation.py
```

**Что проверяется:**
- ✅ Целостность данных
- ✅ Пропущенные значения
- ✅ Дубликаты
- ✅ Типы данных
- ✅ Корреляция признаков
- ✅ Оценка модели

**Результаты:** `reports/deepchecks_*.html`

### 2. Мониторинг дрейфа (EvidentlyAI)

```bash
cd src
python evidently_monitoring.py
```

**Что анализируется:**
- 📊 Дрейф распределения признаков
- 📊 Дрейф целевой переменной
- 📊 Качество данных
- 📊 Производительность модели
- 📊 Статистические тесты

**Результаты:** `reports/evidently_*.html`

### 3. Обучение модели (MLflow)

```bash
cd src

# Random Forest (по умолчанию)
python train.py --model-type rf --n-estimators 100 --max-depth 10

# Linear Regression
python train.py --model-type lr
```

**Что логируется:**
- 📈 Параметры модели
- 📈 Метрики (MSE, RMSE, MAE, R²)
- 📈 Артефакты (модель, предсказания)
- 📈 Метаданные (дата, версия, автор)

**Результаты:**
- Модель: `models/rf_model.pkl`
- MLflow: `mlruns/`
- Предсказания: `reports/predictions.csv`

### 4. Просмотр экспериментов

```bash
mlflow ui
```

Откройте браузер: http://localhost:5000

## 🔄 CI/CD

### GitHub Actions

Автоматически запускается при:
- Push в `main` или `develop`
- Pull Request в `main`
- По расписанию (каждый день в 00:00 UTC)

**Этапы:**
1. ✅ Test - линтинг и тесты
2. ✅ Data Validation - Deepchecks
3. ✅ Drift Monitoring - EvidentlyAI
4. ✅ Train Model - обучение с MLflow
5. ✅ Deploy - деплой модели

**Просмотр:** https://github.com/your-username/ml-pipeline-project/actions

### GitLab CI

**Этапы:**
- install - установка зависимостей
- test - тесты и линтинг
- validate - валидация данных
- train - обучение модели
- deploy - деплой (manual trigger)

**Просмотр:** https://gitlab.com/your-username/ml-pipeline-project/-/pipelines

## 🐳 Docker

### Сборка образа

```bash
docker build -t ml-pipeline:latest .
```

### Запуск контейнера

```bash
# MLflow UI
docker run -p 5000:5000 ml-pipeline:latest

# Обучение модели
docker run ml-pipeline:latest python src/train.py

# Валидация данных
docker run ml-pipeline:latest python src/deepchecks_validation.py

# Мониторинг дрейфа
docker run ml-pipeline:latest python src/evidently_monitoring.py
```

### Docker Compose (опционально)

```bash
docker-compose up
```

## 📖 Детальная инструкция

### Этап 1: Подготовка окружения

#### 1.1 Создание аккаунтов

1. **GitHub:**
   - Перейдите на https://github.com/signup
   - Создайте аккаунт
   - Создайте новый репозиторий

2. **GitLab:**
   - Перейдите на https://gitlab.com/users/sign_up
   - Создайте аккаунт
   - Создайте новый проект

#### 1.2 Настройка Git

```bash
# Инициализация репозитория
git init
git add .
git commit -m "Initial commit: ML Pipeline project"

# GitHub
git remote add origin https://github.com/your-username/ml-pipeline-project.git
git push -u origin main

# GitLab
git remote add gitlab https://gitlab.com/your-username/ml-pipeline-project.git
git push gitlab main
```

#### 1.3 Настройка Git LFS (для больших файлов)

```bash
# Установка Git LFS
git lfs install

# Трекинг файлов (уже настроено в .gitattributes)
git lfs track "*.pkl"
git lfs track "*.h5"

# Коммит
git add .gitattributes
git commit -m "Add Git LFS configuration"
git push
```

### Этап 2: Проверка данных с Deepchecks

#### 2.1 Запуск проверки

```bash
cd src
python deepchecks_validation.py
```

#### 2.2 Анализ результатов

Откройте в браузере:
- `reports/deepchecks_data_integrity.html` - целостность данных
- `reports/deepchecks_model_evaluation.html` - оценка модели

**Что проверяется:**
- Пропущенные значения
- Дубликаты строк и колонок
- Смешанные типы данных
- Одинаковые значения
- Специальные символы
- Корреляция признаков с целевой переменной

#### 2.3 Интерпретация

- ✅ **Passed** - проверка пройдена
- ⚠️ **Warning** - есть проблемы, но некритичные
- ❌ **Failed** - критические проблемы

### Этап 3: Мониторинг дрейфа с EvidentlyAI

#### 3.1 Запуск мониторинга

```bash
cd src
python evidently_monitoring.py
```

#### 3.2 Анализ результатов

Откройте в браузере:
- `evidently_data_drift.html` - дрейф данных
- `evidently_data_quality.html` - качество данных
- `evidently_model_performance.html` - производительность модели
- `evidently_drift_tests.html` - тесты на дрейф
- `evidently_data_drift_artificial.html` - искусственный дрейф (для демонстрации)

**Метрики дрейфа:**
- **Data Drift Score** - общий показатель дрейфа
- **Share of Drifted Features** - доля признаков с дрейфом
- **Target Drift** - дрейф целевой переменной

#### 3.3 Что делать при дрейфе?

- Переобучить модель на новых данных
- Добавить новые признаки
- Изменить preprocessing
- Настроить пороги алертов

### Этап 4: Обучение модели с MLflow

#### 4.1 Запуск обучения

```bash
cd src

# С параметрами по умолчанию
python train.py

# С кастомными параметрами
python train.py --model-type rf --n-estimators 200 --max-depth 15
```

#### 4.2 Просмотр экспериментов

```bash
mlflow ui
```

Откройте: http://localhost:5000

**Что можно сделать:**
- Сравнить эксперименты
- Посмотреть метрики и параметры
- Скачать артефакты
- Зарегистрировать лучшую модель

#### 4.3 Использование MLflow Projects

```bash
# Запуск полного pipeline
mlflow run . -e full_pipeline

# Только обучение
mlflow run . -e main -P model_type=rf -P n_estimators=100

# Только валидация
mlflow run . -e validate_data
```

### Этап 5: Автоматизация с GitHub Actions

#### 5.1 Настройка секретов (опционально)

В настройках репозитория на GitHub:
1. Settings → Secrets → Actions
2. Добавьте секрет `MLFLOW_TRACKING_URI` (если используется внешний MLflow)

#### 5.2 Просмотр результатов

1. Откройте вкладку "Actions" в репозитории
2. Выберите workflow run
3. Посмотрите логи каждого job
4. Скачайте артефакты

#### 5.3 Локальный запуск GitHub Actions (act)

```bash
# Установка act
brew install act  # Mac
# или скачайте с https://github.com/nektos/act

# Запуск локально
act -j test
act -j train-model
```

### Этап 6: Автоматизация с GitLab CI

#### 6.1 Настройка переменных (опционально)

В настройках проекта на GitLab:
1. Settings → CI/CD → Variables
2. Добавьте переменные окружения

#### 6.2 Просмотр результатов

1. Откройте CI/CD → Pipelines
2. Выберите pipeline
3. Посмотрите логи каждого job
4. Скачайте артефакты

#### 6.3 Просмотр GitLab Pages

После деплоя откройте:
https://your-username.gitlab.io/ml-pipeline-project/

### Этап 7: Тестирование

#### 7.1 Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С coverage
pytest tests/ --cov=src --cov-report=html

# Конкретный тест
pytest tests/test_pipeline.py::test_data_loading -v
```

#### 7.2 Просмотр coverage

Откройте `htmlcov/index.html` в браузере

### Этап 8: Деплой модели (опционально)

#### 8.1 Регистрация модели в MLflow

```python
import mlflow

model_uri = "runs:/<run_id>/model"
mlflow.register_model(model_uri, "CaliforniaHousingModel")
```

#### 8.2 Продакшн версия

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="CaliforniaHousingModel",
    version=1,
    stage="Production"
)
```

## 📊 Метрики и мониторинг

### MLflow Metrics
- **MSE** (Mean Squared Error) - средняя квадратичная ошибка
- **RMSE** (Root Mean Squared Error) - корень из MSE
- **MAE** (Mean Absolute Error) - средняя абсолютная ошибка
- **R²** (Coefficient of Determination) - коэффициент детерминации

### Evidently Metrics
- **Data Drift Score** - показатель дрейфа данных
- **Feature Drift** - дрейф каждого признака
- **Target Drift** - дрейф целевой переменной
- **Data Quality Score** - качество данных

## 🔧 Решение проблем

### Проблема: MLflow не запускается

```bash
# Проверьте порт
lsof -i :5000

# Убейте процесс
kill -9 <PID>

# Перезапустите
mlflow ui --port 5001
```

### Проблема: Ошибки импорта

```bash
# Переустановите зависимости
pip install -r requirements.txt --force-reinstall

# Или создайте новое окружение
python -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

### Проблема: Git LFS ошибки

```bash
# Установите Git LFS
git lfs install

# Загрузите файлы
git lfs pull

# Проверьте tracked файлы
git lfs ls-files
```

## 📚 Дополнительные ресурсы

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Deepchecks Documentation](https://docs.deepchecks.com/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI Documentation](https://docs.gitlab.com/ee/ci/)

## 👨‍💻 Автор

**Sergey** - PhD кандидат, МФТИ
- Специализация: Machine Learning, Криптография
- Проект: Защищенный семантический поиск с гомоморфным шифрованием

## 📄 Лицензия

MIT License

## 🤝 Вклад

Pull requests приветствуются! Для серьезных изменений сначала откройте issue.
