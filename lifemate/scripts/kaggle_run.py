"""Đẩy notebook lên Kaggle (GPU T4 x2), chờ chạy xong, tải output về output/<RUN_TAG>/.

Dùng:
  .venv/bin/python scripts/kaggle_run.py --set SPLIT=train LIMIT=30 RUN_TAG=v2
  .venv/bin/python scripts/kaggle_run.py --set SPLIT=public_test LIMIT=None RUN_TAG=v2 --resume
  .venv/bin/python scripts/kaggle_run.py --fetch-only RUN_TAG=v2      (chỉ tải output của version mới nhất)

  .venv/bin/python scripts/kaggle_run.py --nb asr --set LIMIT=20 RUN_TAG=asr1   (notebook so sánh ASR)
  .venv/bin/python scripts/kaggle_run.py --source jasonhoang666/zalo-ai-asr --set 'TRANSCRIPT_SRC="asr_vietasr.json"' ...

--set ghi đè giá trị trong dict CFG của notebook (giá trị viết như Python: "train", 30, None, True).
--resume gắn output của chính kernel này làm Input để notebook chạy tiếp từ checkpoint.
--source gắn output của kernel khác làm Input (vd transcript của zalo-ai-asr).
RUN_TAG chỉ dùng đặt tên thư mục output/<RUN_TAG>/ khi CFG của notebook không có khoá này.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {
    "pipeline": (ROOT / "kaggle", ROOT / "notebooks" / "lifemate_pipeline.py"),
    # kernel thứ hai cho cùng notebook: `kernels status` chỉ báo version mới nhất, nên hai run song song phải ở hai kernel
    "pipeline2": (ROOT / "kaggle_b", ROOT / "notebooks" / "lifemate_pipeline.py"),
    "asr": (ROOT / "kaggle_asr", ROOT / "notebooks" / "asr_bench.py"),
    "lora": (ROOT / "kaggle_asr_lora", ROOT / "notebooks" / "asr_lora.py"),
    # cùng kernel với "lora": chạy với --resume để nạp adapter từ output trước của chính kernel
    "lora_infer": (ROOT / "kaggle_asr_lora", ROOT / "notebooks" / "asr_lora_infer.py"),
    "asr_prompt": (ROOT / "kaggle_asr_lora", ROOT / "notebooks" / "asr_prompt.py"),
    "sft": (ROOT / "kaggle_sft", ROOT / "notebooks" / "llm_sft.py"),
}
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def sh(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} lỗi:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def build_notebook(kdir, src_path, overrides):
    src = src_path.read_text(encoding="utf-8")
    for key, val in overrides.items():
        src, n = re.subn(rf'^(\s*"{key}":\s*)[^,]+,', lambda m: f"{m.group(1)}{val},", src, count=1, flags=re.M)
        if n == 0 and key != "RUN_TAG":
            sys.exit(f"Không thấy khoá CFG '{key}' trong {src_path.name}")
    tmp = kdir / src_path.name
    tmp.write_text(src, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "py_to_ipynb.py"), str(tmp)], check=True)
    tmp.unlink()


def wait(kernel, poll_s=60, max_bad=5):
    t0, bad = time.time(), 0
    while True:
        r = subprocess.run([KAGGLE, "kernels", "status", kernel], capture_output=True, text=True)
        m = re.search(r'status "?([A-Za-z_.]+)', r.stdout)
        status = m.group(1).lower() if m else ""
        print(f"[{(time.time() - t0) / 60:5.1f} phút] {status or (r.stdout + r.stderr).strip()[:200]}", flush=True)
        if any(s in status for s in ("complete", "error", "cancel")):
            return status
        # CLI mất quyền (token hết hạn) thì status rỗng mãi; dừng thay vì chờ vô hạn
        bad = 0 if status else bad + 1
        if bad >= max_bad:
            sys.exit(f"Không đọc được trạng thái {kernel} {max_bad} lần liền; kiểm tra `kaggle auth login`")
        time.sleep(poll_s)


def fetch(kernel, tag, version=None):
    dest = ROOT / "output" / tag
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # không ghi version thì CLI lấy version mới nhất, có thể là của lần push khác lên cùng kernel
    # trọng số model (safetensors, vài GB) để lại trên Kaggle, chỉ tải log và kết quả về máy
    sh("kernels", "output", f"{kernel}/{version}" if version else kernel, "-p", str(dest), "-o",
       "--file-pattern", r"^(?!.*\.safetensors$).*$")
    print(f"Output: {dest}")
    for p in sorted(dest.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dest)}  {p.stat().st_size // 1024} KB")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fetch-only", metavar="RUN_TAG")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--nb", choices=NOTEBOOKS, default="pipeline")
    ap.add_argument("--source", nargs="*", default=[], metavar="OWNER/KERNEL")
    ap.add_argument("--dataset", nargs="*", default=[], metavar="OWNER/DATASET")
    args = ap.parse_args()

    kdir, src_path = NOTEBOOKS[args.nb]
    meta_path = kdir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kernel = meta["id"]

    if args.fetch_only:
        fetch(kernel, args.fetch_only)
        return

    overrides = dict(kv.split("=", 1) for kv in args.set)
    tag = overrides.get("RUN_TAG", "run").strip("\"'")
    build_notebook(kdir, src_path, overrides)
    # một kernel có thể chạy nhiều notebook nguồn (lora / lora_infer), nên code_file phải theo notebook vừa build
    meta["code_file"] = src_path.with_suffix(".ipynb").name
    meta["kernel_sources"] = ([kernel] if args.resume else []) + args.source
    # dataset gắn thêm được giữ lại trong metadata cho các lần push sau; vô hại vì model chọn theo tên thư mục
    meta["dataset_sources"] += [d for d in args.dataset if d not in meta["dataset_sources"]]
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out = sh("kernels", "push", "-p", str(kdir))
    print(out)
    # CLI trả exit 0 cả khi push bị từ chối (vd hết suất GPU), nên phải đọc nội dung
    if "push error" in out.lower():
        sys.exit(1)
    if args.no_wait:
        return
    version = re.search(r"version (\d+)", out)
    time.sleep(30)
    status = wait(kernel)
    dest = fetch(kernel, tag, version.group(1) if version else None)
    if "complete" not in status:
        print(f"Kernel kết thúc với trạng thái {status}; xem log trong {dest}")


if __name__ == "__main__":
    main()
