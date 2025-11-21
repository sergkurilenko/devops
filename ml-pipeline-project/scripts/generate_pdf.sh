#!/bin/bash
# Скрипт для генерации PDF из markdown документа

echo "Генерация PDF из DEPLOYMENT_GUIDE.md"

# Проверка наличия pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Ошибка: pandoc не установлен"
    echo "Установите pandoc:"
    echo "  Ubuntu/Debian: sudo apt-get install pandoc texlive-latex-base"
    echo "  Mac: brew install pandoc basictex"
    echo "  Windows: choco install pandoc miktex"
    exit 1
fi

# Генерация PDF
pandoc DEPLOYMENT_GUIDE.md \
    -o DEPLOYMENT_GUIDE.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V documentclass=article \
    -V lang=ru \
    --toc \
    --toc-depth=3 \
    --highlight-style=tango

if [ $? -eq 0 ]; then
    echo "✓ PDF успешно создан: DEPLOYMENT_GUIDE.pdf"
    ls -lh DEPLOYMENT_GUIDE.pdf
else
    echo "✗ Ошибка при создании PDF"
    echo ""
    echo "Альтернативный метод (без LaTeX):"
    echo "  pip install grip"
    echo "  grip DEPLOYMENT_GUIDE.md --export DEPLOYMENT_GUIDE.html"
    echo "  # Затем откройте HTML в браузере и сохраните как PDF"
fi
