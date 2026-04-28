import os
from modelscope.hub.snapshot_download import snapshot_download


def download_qwen3_model():
    print("🚀 Starting Qwen3-0.6B model download (ModelScope)...")
    print("=" * 50)

    target_dir = "./models/Qwen3-0.6B"

    try:
        print("📦 Downloading from ModelScope...")

        cache_dir = os.path.abspath("./models/.cache")

        model_cache_path = snapshot_download(
            model_id="Qwen/Qwen3-0.6B",
            cache_dir=cache_dir
        )

        model_cache_path = os.path.abspath(model_cache_path)

        print(f"📁 Raw model path: {model_cache_path}")

        if os.path.islink(target_dir) or os.path.exists(target_dir):
            print("⚠️ Removing existing target_dir...")
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
    success = download_qwen3_model()
    exit(0 if success else 1)