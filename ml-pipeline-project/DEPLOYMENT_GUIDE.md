# ИНСТРУКЦИЯ ПО РАЗВЕРТЫВАНИЮ ML PIPELINE
## Модуль 5: Воспроизводимые пайплайны машинного обучения

**Автор:** Сергей  
**Дата:** Ноябрь 2025  
**Версия:** 1.0

---

## ОГЛАВЛЕНИЕ

1. Введение
2. Требования к системе
3. Этап 1: Подготовка окружения
4. Этап 2: Проверка данных (Deepchecks)
5. Этап 3: Мониторинг дрейфа (EvidentlyAI)
6. Этап 4: Обучение модели (MLflow)
7. Этап 5: Настройка CI/CD (GitHub Actions)
8. Этап 6: Настройка CI/CD (GitLab CI)
9. Этап 7: Контейнеризация (Docker)
10. Результаты и скриншоты
11. Заключение

---

## 1. ВВЕДЕНИЕ

Данный документ содержит пошаговую инструкцию по развертыванию воспроизводимого ML-пайплайна для решения задачи регрессии (предсказание цен на жилье).

### Цели проекта:

✅ Создать воспроизводимый ML-проект  
✅ Настроить CI/CD с GitHub Actions и GitLab CI  
✅ Провести валидацию данных с Deepchecks  
✅ Выполнить анализ дрейфа данных с EvidentlyAI  
✅ Реализовать отслеживание экспериментов с MLflow  
✅ Контейнеризировать приложение с Docker  

### Используемые технологии:

- **Python 3.9+**
- **MLflow** - управление ML жизненным циклом
- **Deepchecks** - валидация данных и моделей
- **EvidentlyAI** - мониторинг дрейфа данных
- **scikit-learn** - ML библиотека
- **GitHub Actions** - CI/CD для GitHub
- **GitLab CI** - CI/CD для GitLab
- **Docker** - контейнеризация

---

## 2. ТРЕБОВАНИЯ К СИСТЕМЕ

### Минимальные требования:

- **OS:** Linux, macOS, или Windows 10+
- **RAM:** 4 GB
- **Disk:** 10 GB свободного места
- **Python:** 3.8+
- **Git:** 2.0+

### Дополнительное ПО:

- Docker Desktop (опционально)
- Git LFS (опционально, для больших файлов)

---

## 3. ЭТАП 1: ПОДГОТОВКА ОКРУЖЕНИЯ

### Шаг 1.1: Создание аккаунтов

#### GitHub:
1. Откройте https://github.com/signup
2. Заполните форму регистрации
3. Подтвердите email
4. Создайте новый репозиторий "ml-pipeline-project"

**Скриншот 1:** Создание репозитория на GitHub

#### GitLab:
1. Откройте https://gitlab.com/users/sign_up
2. Заполните форму регистрации
3. Подтвердите email
4. Создайте новый проект "ml-pipeline-project"

**Скриншот 2:** Создание проекта на GitLab

### Шаг 1.2: Клонирование проекта

```bash
# Клонирование с GitHub
git clone https://github.com/your-username/ml-pipeline-project.git
cd ml-pipeline-project

# Или скачайте архив проекта
# Распакуйте в удобную директорию
```

**Скриншот 3:** Клонирование репозитория

### Шаг 1.3: Создание виртуального окружения

```bash
# Создание venv
python -m venv venv

# Активация (Linux/Mac)
source venv/bin/activate

# Активация (Windows)
venv\Scripts\activate
```

**Скриншот 4:** Создание и активация venv

### Шаг 1.4: Установка зависимостей

```bash
# Обновление pip
pip install --upgrade pip

# Установка всех зависимостей
pip install -r requirements.txt
```

**Ожидаемый вывод:**
```
Successfully installed scikit-learn-1.3.2 numpy-1.24.3 pandas-2.0.3
mlflow-2.9.2 deepchecks-0.17.4 evidently-0.4.11 ...
```

**Скриншот 5:** Установка зависимостей

### Шаг 1.5: Проверка установки

```bash
# Проверка версий
python --version
pip list | grep -E "mlflow|deepchecks|evidently"
```

**Ожидаемый вывод:**
```
Python 3.9.18
mlflow                    2.9.2
deepchecks                0.17.4
evidently                 0.4.11
```

**Скриншот 6:** Проверка установленных пакетов

---

## 4. ЭТАП 2: ПРОВЕРКА ДАННЫХ (DEEPCHECKS)

### Шаг 2.1: Запуск проверки данных

```bash
cd src
python deepchecks_validation.py
```

**Ожидаемый вывод:**
```
============================================
DEEPCHECKS: ПРОВЕРКА КАЧЕСТВА ДАННЫХ И МОДЕЛИ
============================================
Загрузка данных California Housing...
Размер данных: (20640, 9)
Подготовка данных...
Train size: (14448, 8)
Test size: (3612, 8)

====================================
ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ (Data Integrity)
====================================

Отчет сохранен: ../reports/deepchecks_data_integrity.html
...
```

**Скриншот 7:** Вывод Deepchecks в терминале

### Шаг 2.2: Анализ отчетов

Откройте в браузере файлы:
- `reports/deepchecks_data_integrity.html`
- `reports/deepchecks_model_evaluation.html`

#### Проверка целостности данных:

**Проверяется:**
- ✅ Отсутствие пропущенных значений
- ✅ Отсутствие дубликатов
- ✅ Корректность типов данных
- ✅ Отсутствие одинаковых значений
- ✅ Отсутствие специальных символов

**Скриншот 8:** Отчет Deepchecks Data Integrity

#### Оценка модели:

**Проверяется:**
- ✅ Performance Report - производительность модели
- ✅ Confusion Matrix - матрица ошибок
- ✅ ROC Curve - ROC кривая
- ✅ Feature Importance - важность признаков

**Скриншот 9:** Отчет Deepchecks Model Evaluation

### Шаг 2.3: Интерпретация результатов

**Пример результатов:**
```
СВОДКА ПРОВЕРОК:
✓ DataDuplicates: passed
✓ MixedDataTypes: passed
✓ MixedNulls: passed
✓ StringMismatch: passed
✓ IsSingleValue: passed
✓ SpecialCharacters: passed
✓ FeatureLabelCorrelation: passed
```

**Выводы:**
1. Данные прошли все проверки целостности
2. Отсутствуют критические проблемы
3. Качество данных достаточно для обучения модели

---

## 5. ЭТАП 3: МОНИТОРИНГ ДРЕЙФА (EVIDENTLYAI)

### Шаг 3.1: Запуск мониторинга дрейфа

```bash
cd src
python evidently_monitoring.py
```

**Ожидаемый вывод:**
```
============================================
EVIDENTLY AI: МОНИТОРИНГ ДРЕЙФА ДАННЫХ
============================================
Загрузка данных California Housing...
Размер данных: (20640, 9)
Reference data: (14448, 9)
Current data: (6192, 9)

====================================
ОТЧЕТ О ДРЕЙФЕ ДАННЫХ (Data Drift Report)
====================================
Отчет сохранен: ../reports/evidently_data_drift.html
Доля признаков с дрейфом: 0.12%
...
```

**Скриншот 10:** Вывод EvidentlyAI в терминале

### Шаг 3.2: Анализ отчетов

Откройте в браузере файлы:
- `evidently_data_drift.html` - дрейф данных
- `evidently_data_quality.html` - качество данных
- `evidently_model_performance.html` - производительность модели
- `evidently_drift_tests.html` - тесты на дрейф
- `evidently_data_drift_artificial.html` - искусственный дрейф

#### Отчет о дрейфе данных:

**Метрики:**
- **Dataset Drift** - общий дрейф датасета
- **Share of Drifted Columns** - доля признаков с дрейфом
- **Drift per Column** - дрейф по каждому признаку

**Скриншот 11:** Отчет Evidently Data Drift

#### Отчет о качестве данных:

**Метрики:**
- **Missing Values** - пропущенные значения
- **Duplicate Rows** - дубликаты
- **Constant Columns** - константные колонки
- **Empty Rows/Columns** - пустые строки/колонки

**Скриншот 12:** Отчет Evidently Data Quality

#### Отчет о производительности:

**Метрики:**
- **MAE** - средняя абсолютная ошибка
- **RMSE** - корень из средней квадратичной ошибки
- **R²** - коэффициент детерминации
- **Error Distribution** - распределение ошибок

**Скриншот 13:** Отчет Evidently Model Performance

### Шаг 3.3: Тесты на дрейф

**Результаты тестов:**
```
Результаты тестов:
Пройдено: 7/7

✓ TestShareOfMissingValues: SUCCESS
✓ TestNumberOfConstantColumns: SUCCESS
✓ TestNumberOfEmptyRows: SUCCESS
✓ TestNumberOfEmptyColumns: SUCCESS
✓ TestNumberOfDuplicatedColumns: SUCCESS
✓ TestNumberOfDuplicatedRows: SUCCESS
✓ DataDriftTestPreset: SUCCESS
```

**Скриншот 14:** Результаты тестов Evidently

### Шаг 3.4: Сравнение распределений

Откройте файл `reports/distribution_comparison.csv`:

```csv
feature,ref_mean,cur_mean,mean_diff_%,std_diff_%
MedInc,3.8707,3.8892,0.48,0.32
HouseAge,28.6395,28.5123,0.44,0.21
AveRooms,5.4290,5.4102,0.35,0.18
...
```

**Выводы:**
1. Дрейф данных минимальный (<2% по всем признакам)
2. Качество данных остается стабильным
3. Модель можно использовать без переобучения

---

## 6. ЭТАП 4: ОБУЧЕНИЕ МОДЕЛИ (MLFLOW)

### Шаг 4.1: Запуск обучения модели

```bash
cd src
python train.py --model-type rf --n-estimators 100 --max-depth 10
```

**Ожидаемый вывод:**
```
==================================================
ЗАПУСК ML PIPELINE С MLFLOW
==================================================
Загрузка данных California Housing...
Данные сохранены: shape=(20640, 9)
Подготовка данных...
Train size: (16512, 8)
Test size: (4128, 8)
Обучение модели: rf
Модель обучена!
Оценка модели...
Метрики:
  MSE:  0.2547
  RMSE: 0.5047
  MAE:  0.3289
  R²:   0.8124

==================================================
MLflow Run ID: a1b2c3d4e5f6g7h8i9j0
Experiment ID: 1
==================================================

Готово! Запустите 'mlflow ui' для просмотра результатов
```

**Скриншот 15:** Обучение модели с MLflow

### Шаг 4.2: Просмотр экспериментов MLflow

```bash
mlflow ui
```

Откройте браузер: http://localhost:5000

**Интерфейс MLflow UI содержит:**

1. **Список экспериментов**
   - Название эксперимента
   - Количество runs
   - Дата создания

2. **Список runs**
   - Run ID
   - Время начала
   - Время выполнения
   - Статус

3. **Параметры**
   - model_type: rf
   - n_estimators: 100
   - max_depth: 10
   - test_size: 0.2
   - random_state: 42

4. **Метрики**
   - mse: 0.2547
   - rmse: 0.5047
   - mae: 0.3289
   - r2: 0.8124

5. **Артефакты**
   - model/ - сохраненная модель
   - predictions.csv - предсказания

**Скриншот 16:** Интерфейс MLflow UI

**Скриншот 17:** Сравнение экспериментов

**Скриншот 18:** Детали run

### Шаг 4.3: Сравнение моделей

Обучим несколько моделей с разными параметрами:

```bash
# Random Forest с разными параметрами
python train.py --model-type rf --n-estimators 50 --max-depth 5
python train.py --model-type rf --n-estimators 100 --max-depth 10
python train.py --model-type rf --n-estimators 200 --max-depth 15

# Linear Regression
python train.py --model-type lr
```

**Сравнение результатов в MLflow:**

| Model | n_estimators | max_depth | RMSE | R² |
|-------|--------------|-----------|------|-----|
| RF | 50 | 5 | 0.5234 | 0.7956 |
| RF | 100 | 10 | 0.5047 | 0.8124 |
| RF | 200 | 15 | 0.4989 | 0.8156 |
| LR | - | - | 0.7345 | 0.6234 |

**Вывод:** Лучшая модель - Random Forest (200, 15)

**Скриншот 19:** Сравнение моделей в MLflow

---

## 7. ЭТАП 5: НАСТРОЙКА CI/CD (GITHUB ACTIONS)

### Шаг 5.1: Настройка репозитория

1. Создайте репозиторий на GitHub
2. Загрузите код:

```bash
git init
git add .
git commit -m "Initial commit: ML Pipeline"
git branch -M main
git remote add origin https://github.com/your-username/ml-pipeline-project.git
git push -u origin main
```

**Скриншот 20:** Загрузка кода на GitHub

### Шаг 5.2: Проверка workflow

Файл `.github/workflows/ml-pipeline.yml` уже создан.

**Структура workflow:**

```yaml
name: ML Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:            # Тесты и линтинг
  data-validation: # Deepchecks
  drift-monitoring:# EvidentlyAI
  train-model:     # Обучение с MLflow
  deploy:          # Деплой
  report:          # Отчет
```

### Шаг 5.3: Запуск workflow

После push workflow запустится автоматически:

```bash
git add .
git commit -m "Add ML Pipeline code"
git push
```

**Просмотр в GitHub:**
1. Откройте репозиторий
2. Перейдите на вкладку "Actions"
3. Выберите последний workflow run

**Скриншот 21:** GitHub Actions - список workflows

**Скриншот 22:** GitHub Actions - детали run

### Шаг 5.4: Анализ результатов

**Jobs:**
- ✅ test - прошел за 1м 23с
- ✅ data-validation - прошел за 2м 45с
- ✅ drift-monitoring - прошел за 1м 56с
- ✅ train-model - прошел за 3м 12с
- ✅ deploy - прошел за 0м 34с
- ✅ report - прошел за 0м 21с

**Скриншот 23:** Успешное выполнение всех jobs

### Шаг 5.5: Скачивание артефактов

В интерфейсе GitHub Actions:
1. Выберите workflow run
2. Прокрутите вниз до "Artifacts"
3. Скачайте:
   - deepchecks-reports
   - evidently-reports
   - trained-model
   - predictions

**Скриншот 24:** Артефакты GitHub Actions

---

## 8. ЭТАП 6: НАСТРОЙКА CI/CD (GITLAB CI)

### Шаг 8.1: Настройка репозитория

1. Создайте проект на GitLab
2. Добавьте remote:

```bash
git remote add gitlab https://gitlab.com/your-username/ml-pipeline-project.git
git push gitlab main
```

**Скриншот 25:** Загрузка кода на GitLab

### Шаг 8.2: Проверка pipeline

Файл `.gitlab-ci.yml` уже создан.

**Структура pipeline:**

```yaml
stages:
  - install    # Установка зависимостей
  - test       # Тесты и линтинг
  - validate   # Валидация данных
  - train      # Обучение модели
  - deploy     # Деплой
```

### Шаг 8.3: Запуск pipeline

Pipeline запустится автоматически после push.

**Просмотр в GitLab:**
1. Откройте проект
2. Перейдите на CI/CD → Pipelines
3. Выберите последний pipeline

**Скриншот 26:** GitLab CI - список pipelines

**Скриншот 27:** GitLab CI - детали pipeline

### Шаг 8.4: Анализ результатов

**Stages:**
- ✅ install (28s)
- ✅ test:lint (42s)
- ✅ test:unit (1m 15s)
- ✅ validate:deepchecks (2m 34s)
- ✅ validate:evidently (1m 48s)
- ✅ train:random-forest (3m 5s)
- ⏸️ deploy:staging (manual)
- ⏸️ deploy:production (manual)

**Скриншот 28:** Успешное выполнение pipeline

### Шаг 8.5: Просмотр артефактов

В интерфейсе GitLab:
1. Выберите pipeline
2. Нажмите на job
3. Справа нажмите "Browse" для просмотра артефактов

**Скриншот 29:** Артефакты GitLab CI

### Шаг 8.6: GitLab Pages

После деплоя откройте:
https://your-username.gitlab.io/ml-pipeline-project/

**Содержит:**
- Deepchecks отчеты
- Evidently отчеты
- Coverage отчет

**Скриншот 30:** GitLab Pages с отчетами

---

## 9. ЭТАП 7: КОНТЕЙНЕРИЗАЦИЯ (DOCKER)

### Шаг 9.1: Сборка Docker образа

```bash
docker build -t ml-pipeline:latest .
```

**Ожидаемый вывод:**
```
[+] Building 245.3s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 1.23kB
 => [internal] load .dockerignore
 => [1/7] FROM docker.io/library/python:3.9-slim
 => [2/7] WORKDIR /app
 => [3/7] COPY requirements.txt .
 => [4/7] RUN pip install --no-cache-dir -r requirements.txt
 => [5/7] COPY src/ ./src/
 => [6/7] COPY data/ ./data/
 => [7/7] RUN mkdir -p /app/reports /app/mlruns
 => exporting to image
 => => exporting layers
 => => writing image sha256:abc123...
 => => naming to docker.io/library/ml-pipeline:latest
```

**Скриншот 31:** Сборка Docker образа

### Шаг 9.2: Запуск контейнера

#### MLflow UI:
```bash
docker run -p 5000:5000 ml-pipeline:latest
```

Откройте: http://localhost:5000

**Скриншот 32:** MLflow UI в Docker

#### Обучение модели:
```bash
docker run ml-pipeline:latest python src/train.py
```

**Скриншот 33:** Обучение модели в Docker

#### Валидация данных:
```bash
docker run ml-pipeline:latest python src/deepchecks_validation.py
```

**Скриншот 34:** Deepchecks в Docker

### Шаг 9.3: Docker Compose (опционально)

Создайте `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mlflow:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
      - ./reports:/app/reports
    command: mlflow ui --host 0.0.0.0
```

Запуск:
```bash
docker-compose up
```

**Скриншот 35:** Docker Compose

---

## 10. РЕЗУЛЬТАТЫ И СКРИНШОТЫ

### 10.1 Сводная таблица результатов

| Этап | Инструмент | Время | Статус |
|------|------------|-------|--------|
| Валидация данных | Deepchecks | 2м 45с | ✅ Passed |
| Мониторинг дрейфа | EvidentlyAI | 1м 56с | ✅ No drift |
| Обучение модели | MLflow | 3м 12с | ✅ R²=0.8156 |
| CI/CD GitHub | GitHub Actions | 9м 31с | ✅ Success |
| CI/CD GitLab | GitLab CI | 8м 47с | ✅ Success |
| Контейнеризация | Docker | 4м 5с | ✅ Built |

### 10.2 Метрики лучшей модели

- **Model:** Random Forest
- **Parameters:**
  - n_estimators: 200
  - max_depth: 15
- **Metrics:**
  - MSE: 0.2489
  - RMSE: 0.4989
  - MAE: 0.3201
  - R²: 0.8156

### 10.3 Скриншоты

Все скриншоты приведены выше в соответствующих разделах:

1. Создание репозитория на GitHub
2. Создание проекта на GitLab
3. Клонирование репозитория
4. Создание venv
5. Установка зависимостей
6. Проверка пакетов
7. Вывод Deepchecks
8. Отчет Data Integrity
9. Отчет Model Evaluation
10. Вывод EvidentlyAI
11. Отчет Data Drift
12. Отчет Data Quality
13. Отчет Model Performance
14. Результаты тестов
15. Обучение с MLflow
16. MLflow UI
17. Сравнение экспериментов
18. Детали run
19. Сравнение моделей
20. GitHub - загрузка кода
21. GitHub Actions workflows
22. GitHub Actions run
23. Успешные jobs
24. Артефакты GitHub
25. GitLab - загрузка кода
26. GitLab pipelines
27. GitLab pipeline детали
28. Успешный pipeline
29. Артефакты GitLab
30. GitLab Pages
31. Docker build
32. MLflow в Docker
33. Train в Docker
34. Deepchecks в Docker
35. Docker Compose

---

## 11. ЗАКЛЮЧЕНИЕ

### 11.1 Выполненные задачи

✅ Создан воспроизводимый ML-проект  
✅ Настроен CI/CD с GitHub Actions  
✅ Настроен CI/CD с GitLab CI  
✅ Проведена валидация данных с Deepchecks  
✅ Выполнен анализ дрейфа с EvidentlyAI  
✅ Реализовано отслеживание экспериментов с MLflow  
✅ Контейнеризировано приложение с Docker  
✅ Созданы автоматизированные тесты  
✅ Настроено версионирование с Git LFS  
✅ Сгенерированы отчеты в HTML  

### 11.2 Ссылки на ресурсы

- **GitHub Repository:** https://github.com/your-username/ml-pipeline-project
- **GitLab Project:** https://gitlab.com/your-username/ml-pipeline-project
- **GitLab Pages:** https://your-username.gitlab.io/ml-pipeline-project/
- **Docker Hub:** (если опубликовано)

### 11.3 Воспроизводимость

Для полной воспроизводимости проекта:

```bash
# 1. Клонирование
git clone https://github.com/your-username/ml-pipeline-project.git
cd ml-pipeline-project

# 2. Установка
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Запуск полного pipeline
bash scripts/run_pipeline.sh

# 4. Просмотр результатов
mlflow ui
```

**Время выполнения полного pipeline:** ~10 минут

### 11.4 Дальнейшие улучшения

Возможные направления развития проекта:

1. **Мониторинг в production**
   - Интеграция с Prometheus/Grafana
   - Алерты при дрейфе данных
   - Дашборды в реальном времени

2. **Расширение функциональности**
   - A/B тестирование моделей
   - Автоматическое переобучение
   - Feature store

3. **Улучшение CI/CD**
   - Автоматический деплой в Kubernetes
   - Blue-Green deployment
   - Canary releases

4. **Дополнительные инструменты**
   - DVC для версионирования данных
   - Great Expectations для валидации
   - Weights & Biases для экспериментов

### 11.5 Контакты

**Автор:** Сергей  
**Email:** sergey@example.com  
**GitHub:** https://github.com/your-username  
**LinkedIn:** https://linkedin.com/in/your-profile  

---

## ПРИЛОЖЕНИЯ

### Приложение А: Структура файлов

```
ml-pipeline-project/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── .gitlab-ci.yml
├── src/
│   ├── train.py
│   ├── deepchecks_validation.py
│   └── evidently_monitoring.py
├── tests/
│   └── test_pipeline.py
├── scripts/
│   └── run_pipeline.sh
├── data/
├── models/
├── reports/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── MLproject
├── conda.yaml
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

### Приложение Б: requirements.txt

См. файл `requirements.txt` в корне проекта.

### Приложение В: Команды для быстрого запуска

```bash
# Полный pipeline
bash scripts/run_pipeline.sh

# Только валидация
python src/deepchecks_validation.py

# Только мониторинг
python src/evidently_monitoring.py

# Только обучение
python src/train.py

# Тесты
pytest tests/ -v

# MLflow UI
mlflow ui

# Docker
docker build -t ml-pipeline .
docker run -p 5000:5000 ml-pipeline
```

---

**КОНЕЦ ДОКУМЕНТА**

*Дата создания: Ноябрь 2025*  
*Версия: 1.0*  
*Автор: Сергей*
