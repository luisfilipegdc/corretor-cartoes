#!/bin/zsh
# Dê dois cliques para abrir o Corretor de Cartões.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements.txt || { echo "Falha ao instalar. Instale o uv: https://docs.astral.sh/uv/"; read; exit 1; }
fi
(sleep 1.5; open "http://127.0.0.1:5055") &
exec .venv/bin/python app.py
