#!/usr/bin/env bash
# Kiểm tra / cài môi trường local: .venv có kaggle CLI + numpy, rồi kiểm tra đăng nhập Kaggle.
# Chạy từ thư mục lifemate/:  bash scripts/setup_env.sh
# Thoát mã 0 = sẵn sàng, 2 = cần đăng nhập (chạy: .venv/bin/kaggle auth login).
set -euo pipefail
cd "$(dirname "$0")/.."

# kaggle CLI chưa chạy trên Python 3.14+, nên ghim 3.12.
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  else
    py=$(command -v python3.12 || command -v python3.11 || command -v python3)
    "$py" -m venv .venv
  fi
fi

# Bản kaggle hiện tại import six nhưng không khai báo nó là dependency.
# numpy + scikit-learn cho scripts/dev/*_smoke.py (nạp các cell của notebook pipeline).
PKGS="kaggle six numpy scikit-learn"
if ! .venv/bin/kaggle --version >/dev/null 2>&1 || ! .venv/bin/python -c "import numpy, sklearn" >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python $PKGS
  else
    .venv/bin/python -m pip install -q $PKGS
  fi
fi
echo "kaggle CLI: $(.venv/bin/kaggle --version 2>&1 | head -1)"

# Chỉ kiểm tra lệnh chạy được, không đọc hay in token.
if .venv/bin/kaggle kernels list --mine --page-size 1 >/dev/null 2>&1; then
  echo "Kaggle auth: OK"
else
  echo "Kaggle auth: CHƯA ĐĂNG NHẬP hoặc hết hạn -> chạy: .venv/bin/kaggle auth login (tài khoản jasonhoang666)"
  exit 2
fi
