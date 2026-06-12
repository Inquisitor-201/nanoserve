"""
Download Qwen3 models.

Thin wrapper around `hf download` — just prints the command and runs it.
If you prefer, copy-paste the printed command directly.

Usage:
  python download_model.py 0.6b
  python download_model.py 1.7b
  python download_model.py 1.7b-awq
"""

import os
import sys
import subprocess

MODELS = {
    "0.6b":     ("Qwen/Qwen3-0.6B",           "./models/Qwen3-0.6B"),
    "1.7b":     ("Qwen/Qwen3-1.7B",           "./models/Qwen3-1.7B"),
    "1.7b-awq": ("Orion-zhen/Qwen3-1.7B-AWQ", "./models/Qwen3-1.7B-AWQ"),
}


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else "0.6b"
    key = raw.lower().strip().rstrip("b")
    for k, (repo, local_dir) in MODELS.items():
        if k == key or k.rstrip("b") == key:
            break
    else:
        print(f"Unknown model: {raw}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    # Prefer hf-mirror in China; remove HF_ENDPOINT if you want the official hub
    env = {**os.environ}
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cmd = ["hf", "download", repo, "--local-dir", local_dir]
    print(f"+ {' '.join(cmd)}")
    print(f"  (HF_ENDPOINT={env['HF_ENDPOINT']})\n")
    subprocess.check_call(cmd, env=env)
    print(f"\nDone → {local_dir}")


if __name__ == "__main__":
    main()
