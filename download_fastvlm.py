"""
Download the FastVLM-1.5B model snapshot into models/FastVLM-1.5B.

Usage:
    python download_fastvlm.py

Requires network access and the `huggingface-hub` package.
"""

import os
from huggingface_hub import snapshot_download


def main() -> None:
    target_dir = os.path.join("models", "FastVLM-1.5B")
    os.makedirs(target_dir, exist_ok=True)
    # Download a full snapshot so transformers can load from this folder.
    snapshot_download(
        repo_id="apple/FastVLM-1.5B",
        local_dir=target_dir,
        # You can pin a specific revision/tag/commit here if needed.
        # revision="main",
        ignore_patterns=["*.msgpack", "*.h5"],  # keep it lean if extras exist
    )
    print(f"Model snapshot downloaded to: {target_dir}")


if __name__ == "__main__":
    main()
