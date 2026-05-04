#!/usr/bin/env bash
set -euo pipefail

python src/data.py
python src/train.py
python src/evaluate.py

echo "Pipeline completed. Check artifacts/ for outputs."
