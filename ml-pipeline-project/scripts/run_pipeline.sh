#!/bin/bash
# Скрипт для запуска полного ML pipeline

set -e  # Остановка при ошибке

echo "=============================================="
echo "ЗАПУСК ПОЛНОГО ML PIPELINE"
echo "=============================================="

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Переход в корневую директорию проекта
cd "$(dirname "$0")/.."

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Создание виртуального окружения...${NC}"
    python -m venv venv
fi

# Активация виртуального окружения
echo -e "${BLUE}Активация виртуального окружения...${NC}"
source venv/bin/activate

# Установка зависимостей
echo -e "${BLUE}Установка зависимостей...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "=============================================="
echo "ШАГ 1: ПРОВЕРКА ДАННЫХ С DEEPCHECKS"
echo "=============================================="
cd src
python deepchecks_validation.py
cd ..

echo ""
echo "=============================================="
echo "ШАГ 2: МОНИТОРИНГ ДРЕЙФА С EVIDENTLY AI"
echo "=============================================="
cd src
python evidently_monitoring.py
cd ..

echo ""
echo "=============================================="
echo "ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ С MLFLOW"
echo "=============================================="
cd src
python train.py --model-type rf --n-estimators 100 --max-depth 10
cd ..

echo ""
echo -e "${GREEN}=============================================="
echo "PIPELINE ЗАВЕРШЕН УСПЕШНО!"
echo "==============================================${NC}"

echo ""
echo "Сгенерированные артефакты:"
echo "  - Отчеты Deepchecks: reports/deepchecks_*.html"
echo "  - Отчеты Evidently AI: reports/evidently_*.html"
echo "  - Модель: models/rf_model.pkl"
echo "  - MLflow: mlruns/"
echo ""
echo "Для просмотра экспериментов MLflow запустите:"
echo "  mlflow ui"
echo ""
echo "Для просмотра отчетов откройте HTML файлы в браузере"
