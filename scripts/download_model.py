import os
import sys
from modelscope.hub.snapshot_download import snapshot_download


MODEL_MAP = {
    "0.6b": {
        "id": "Qwen/Qwen3-0.6B",
        "dir": "./models/Qwen3-0.6B",
        "desc": "Qwen3-0.6B",
    },
    "1.7b": {
        "id": "Qwen/Qwen3-1.7B",
        "dir": "./models/Qwen3-1.7B",
        "desc": "Qwen3-1.7B",
    },
}


def download_qwen3_model(model_size: str = "0.6b"):
    """
    Download a Qwen3 model from ModelScope.

    Args:
        model_size: Model size, "0.6b" or "1.7b"
    """
    model_info = MODEL_MAP.get(model_size)
    if model_info is None:
        print(f"❌ Unknown model size: {model_size}")
        print(f"   Available: {', '.join(MODEL_MAP.keys())}")
        return False

    model_id = model_info["id"]
    target_dir = model_info["dir"]
    desc = model_info["desc"]

    print(f"🚀 Starting {desc} download (ModelScope)...")
    print("=" * 50)

    try:
        print(f"📦 Downloading from ModelScope (ID: {model_id})...")

        cache_dir = os.path.abspath("./models/.cache")

        model_cache_path = snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir,
        )

        model_cache_path = os.path.abspath(model_cache_path)

        print(f"📁 Raw model path: {model_cache_path}")

        if os.path.islink(target_dir) or os.path.exists(target_dir):
            print("⚠️  Removing existing target_dir...")
            os.system(f"rm -rf {target_dir}")

        print("🔗 Creating directory symlink...")

        os.symlink(model_cache_path, target_dir)

        print(f"✅ Model ready at: {target_dir} -> {model_cache_path}")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        size_arg = sys.argv[1].lower().replace("b", "b").replace(".", "")
        if "0" in size_arg:
            model_size = "0.6b"
        elif "1" in size_arg:
            model_size = "1.7b"
        else:
            print(f"Usage: python download_model.py [0.6b|1.7b]")
            sys.exit(1)
    else:
        model_size = "0.6b"

    success = download_qwen3_model(model_size)
    exit(0 if success else 1)