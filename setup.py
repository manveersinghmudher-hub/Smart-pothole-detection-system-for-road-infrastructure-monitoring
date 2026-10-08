import sys
import zipfile
import subprocess
from pathlib import Path

# --------------------------------------------------
# Install gdown automatically if missing
# --------------------------------------------------
try:
    import gdown
except ImportError:
    print("Installing gdown...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "gdown"]
    )
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


# --------------------------------------------------
# Download helper
# --------------------------------------------------
def download_file(file_id: str, output_path: Path):
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"\nDownloading: {output_path.name}")

    gdown.download(
        url=url,
        output=str(output_path),
        quiet=False,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Failed to download {output_path.name}"
        )

    print(f"Downloaded: {output_path.name}")


# --------------------------------------------------
# Extract ZIP helper
# --------------------------------------------------
def extract_zip(zip_path: Path):
    print(f"\nExtracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(BASE_DIR)

    zip_path.unlink()

    print(f"Deleted: {zip_path.name}")


# --------------------------------------------------
# Download and extract project folders
# --------------------------------------------------
def install_main_folders():

    dfine_folder = BASE_DIR / "D-FINE-OFFICIAL"
    pidnet_folder = BASE_DIR / "PIDNet"

    if not dfine_folder.exists():

        dfine_zip = BASE_DIR / "D-FINE-OFFICIAL.zip"

        download_file(
            DFINE_ZIP_ID,
            dfine_zip
        )

        extract_zip(dfine_zip)

    else:
        print("\n[SKIP] D-FINE-OFFICIAL already exists")

    if not pidnet_folder.exists():

        pidnet_zip = BASE_DIR / "PIDNet.zip"

        download_file(
            PIDNET_ZIP_ID,
            pidnet_zip
        )

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

    dfine_target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pidnet_target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not dfine_target.exists():

        download_file(
            BEST_STG1_ID,
            dfine_target
        )

    else:
        print("\n[SKIP] best_stg1.pth already exists")

    if not pidnet_target.exists():

        download_file(
            PIDNET_MODEL_ID,
            pidnet_target
        )

    else:
        print(
            "\n[SKIP] PIDNet_S_Cityscapes_test.pt already exists"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("=" * 60)
    print("SPDS ASSET INSTALLER")
    print("=" * 60)

    print("\nSTEP 1/2 : Installing project folders")
    install_main_folders()

    if not (BASE_DIR / "D-FINE-OFFICIAL").exists():
        raise RuntimeError(
            "D-FINE-OFFICIAL folder not found after extraction."
        )

    if not (BASE_DIR / "PIDNet").exists():
        raise RuntimeError(
            "PIDNet folder not found after extraction."
        )

    print("\nSTEP 2/2 : Installing checkpoints")
    install_checkpoints()

    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()