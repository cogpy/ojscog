#!/bin/bash
echo "=== Repository Structure Analysis ==="
echo ""
echo "Root directory files:"
ls -1 *.md *.py *.sh *.txt *.log *.json 2>/dev/null | wc -l
echo ""
echo "Key directories:"
find . -maxdepth 1 -type d | grep -v "^\.$" | sort
echo ""
echo "=== Code Quality Checks ==="
echo ""
echo "Python files count:"
find . -name "*.py" -type f | wc -l
echo ""
echo "PHP files count:"
find . -name "*.php" -type f | wc -l
echo ""
echo "JavaScript files count:"
find . -name "*.js" -type f | wc -l
echo ""
echo "=== Documentation Files ==="
find . -maxdepth 1 -name "*.md" -type f | sort
