#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

LOAD_MME_PY="$PROJECT_ROOT/llava/eval/maya/load_mme.py"


cd "$PROJECT_ROOT"

python "$LOAD_MME_PY"

