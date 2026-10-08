import sys
import zipfile
import subprocess
from pathlib import Path

# --------------------------------------------------
# Install required Python packages safely
# --------------------------------------------------
REQUIRED_PACKAGES = [
    "gdown",
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "Pillow",
    "PyYAML",
    "tensorboard",
    # Imported by D-FINE's package initializers and model/config modules.
    "faster-coco-eval",
    "sympy",
    "scipy",
    "psycopg[binary]",
    "calflops",
    # calflops imports its Hugging Face helpers at package import time.
    "transformers",
    "huggingface-hub",
    "accelerate",
]

# Pip distribution names do not always match their Python import names.
IMPORT_NAMES = {
    "opencv-python": "cv2",
    "PyYAML": "yaml",
    "faster-coco-eval": "faster_coco_eval",
    "Pillow": "PIL",
    "huggingface-hub": "huggingface_hub",
    "psycopg[binary]": "psycopg",
}

def install_missing_packages(packages):
    for pkg in packages:
        # Map pip package names to python module import names
        import_name = IMPORT_NAMES.get(pkg, pkg)
        try:
            __import__(import_name)
            print(f"[SKIP] Package '{pkg}' already exists")
        except ImportError:
            print(f"Installing package '{pkg}'...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg]
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to install '{pkg}': {e}")

install_missing_packages(REQUIRED_PACKAGES)

import gdown

# --------------------------------------------------
# Base directory
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Google Drive File IDs
# --------------------------------------------------
DFINE_ZIP_ID = "16URGUkmGet8lI5f2fQ7pceiI-qNfALID"
PIDNET_ZIP_ID = "1iSXovLQXDXlwltXgEHdKlrUdb3kM06tg"

BEST_STG1_ID = "1KAl_I7tOBeoUfz29PxHCNg64TRso5QWI"
PIDNET_MODEL_ID = "17NF9dZj44dLT_8ZNJ6TeU0niMdgB8I2h"
SAMPLE_VIDEO_ID = "1sNnRRm0syT-G5WdypUIauL7XKXuwN_SP"


# --------------------------------------------------
# Download helper with error handling
# --------------------------------------------------
def download_file(file_id: str, output_path: Path):
    if output_path.exists():
        print(f"[SKIP] File '{output_path.name}' already exists")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"\nDownloading: {output_path.name}")

    try:
        gdown.download(
            url=url,
            output=str(output_path),
            quiet=False,
        )
    except Exception as e:
        print(f"[ERROR] Download failed for '{output_path.name}': {e}")
        return

    if not output_path.exists():
        print(f"[ERROR] File '{output_path.name}' was not created.")
    else:
        print(f"Downloaded: {output_path.name}")


# --------------------------------------------------
# Extract ZIP helper with error handling
# --------------------------------------------------
def extract_zip(zip_path: Path):
    if not zip_path.exists():
        print(f"[ERROR] Cannot extract. File '{zip_path.name}' does not exist.")
        return

    print(f"\nExtracting: {zip_path.name}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BASE_DIR)
        zip_path.unlink()
        print(f"Deleted archive: {zip_path.name}")
    except Exception as e:
        print(f"[ERROR] Failed to extract '{zip_path.name}': {e}")


# --------------------------------------------------
# Download and extract project folders
# --------------------------------------------------
def install_main_folders():

    dfine_folder = BASE_DIR / "D-FINE-OFFICIAL"
    pidnet_folder = BASE_DIR / "PIDNet"

    if not dfine_folder.exists():
        dfine_zip = BASE_DIR / "D-FINE-OFFICIAL.zip"
        download_file(DFINE_ZIP_ID, dfine_zip)
        extract_zip(dfine_zip)
    else:
        print("\n[SKIP] D-FINE-OFFICIAL already exists")

    if not pidnet_folder.exists():
        pidnet_zip = BASE_DIR / "PIDNet.zip"
        download_file(PIDNET_ZIP_ID, pidnet_zip)
        extract_zip(pidnet_zip)
    else:
        print("\n[SKIP] PIDNet already exists")


# --------------------------------------------------
# Download checkpoints
# --------------------------------------------------
def install_checkpoints():

    dfine_target = (
        BASE_DIR
        / "D-FINE-OFFICIAL"
        / "output"
        / "pothole_dfine_s_1000"
        / "best_stg1.pth"
    )

    pidnet_target = (
        BASE_DIR
        / "PIDNet"
        / "pretrained_models"
        / "cityscapes"
        / "PIDNet_S_Cityscapes_test.pt"
    )

    dfine_target.parent.mkdir(parents=True, exist_ok=True)
    pidnet_target.parent.mkdir(parents=True, exist_ok=True)

    if not dfine_target.exists():
        download_file(BEST_STG1_ID, dfine_target)
    else:
        print("\n[SKIP] best_stg1.pth already exists")

    if not pidnet_target.exists():
        download_file(PIDNET_MODEL_ID, pidnet_target)
    else:
        print("\n[SKIP] PIDNet_S_Cityscapes_test.pt already exists")


# --------------------------------------------------
# Download sample video into the project root
# --------------------------------------------------
def install_sample_video():
    sample_video = BASE_DIR / "sample-video.mp4"
    download_file(SAMPLE_VIDEO_ID, sample_video)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("=" * 60)
    print("SPDS ASSET INSTALLER")
    print("=" * 60)

    print("\nSTEP 1/3 : Installing project folders")
    install_main_folders()

    if not (BASE_DIR / "D-FINE-OFFICIAL").exists():
        print("[WARNING] D-FINE-OFFICIAL folder missing after step 1.")

    if not (BASE_DIR / "PIDNet").exists():
        print("[WARNING] PIDNet folder missing after step 1.")

    print("\nSTEP 2/3 : Installing checkpoints")
    install_checkpoints()

    print("\nSTEP 3/3 : Installing sample video")
    install_sample_video()

    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
