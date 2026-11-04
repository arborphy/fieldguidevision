"""Run from an authenticated Modal CLI; keeps downloads/results in a volume."""
from pathlib import Path
import subprocess
import sys
import zipfile

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FILES = [
    "organ_benchmark/split_manifests/individual_disjoint.csv",
    "data/embeddings/efficientnet_b0_strict/embeddings.npz",
    "unclip_mvp/groups/clusters_k5.npy",
    "unclip_mvp/groups/clusters_k7.npy",
]
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements(str(HERE / "requirements.txt"))
         .env({"HF_HOME": "/cache/huggingface"})
         .add_local_file(str(HERE / "run.py"), "/demo/unclip_mvp/run.py"))
for relative in FILES:
    # Missing inputs are rejected locally, before invoking the GPU function.
    if (ROOT / relative).is_file():
        image = image.add_local_file(str(ROOT / relative), f"/demo/{relative}")
app = modal.App("fieldguidevision-unclip-mvp")
volume = modal.Volume.from_name("fieldguidevision-unclip-mvp", create_if_missing=True)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=3600,
              volumes={"/cache": volume})
def run_remote(stage: str, decoder: str, revision: str):
    try:
        subprocess.run(["python", "/demo/unclip_mvp/run.py", "--root", "/demo",
                        "--output", "/cache/results", "--stage", stage,
                        "--decoder", decoder, "--decoder-revision", revision, "--fruit-fallback"], check=True)
        if stage == "all":
            result_root = Path('/cache/results')
            with zipfile.ZipFile(result_root / 'demo_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                for path in result_root.rglob('*'):
                    if path.is_file() and path.name != 'demo_results.zip' and 'photos' not in path.parts:
                        z.write(path, path.relative_to(result_root))
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "smoke", decoder: str = "sd2-community/stable-diffusion-2-1-unclip", revision: str = "ac4aaff4a67d57bdab2e9faf277c33d8e49f4df4"):
    if stage not in ("smoke", "all") or not revision:
        raise ValueError("Use --stage smoke|all and --revision <HF commit SHA>")
    subprocess.run([sys.executable, str(HERE / "run.py"), "--stage", "validate"], check=True)
    run_remote.remote(stage, decoder, revision)
    print("Download: modal volume get fieldguidevision-unclip-mvp results ./unclip_results")
