"""Upload accepted pothole detections from the inference CSV to PostgreSQL/RDS."""

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
TABLE_COLUMNS = (
    "detection_id", "device_id", "vehicle_id", "detected_at", "latitude", "longitude",
    "length_cm", "width_cm", "depth_cm", "severity", "confidence", "model_version",
    "image_s3_key",
)
REQUIRED_SETTINGS = ("RDS_HOST", "RDS_PORT", "RDS_DATABASE", "RDS_USERNAME", "RDS_PASSWORD")


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE entries without requiring python-dotenv."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def upload_csv(csv_path: Path, env_path: Path = ENV_PATH) -> int:
    load_env_file(env_path)
    missing = [name for name in REQUIRED_SETTINGS if not os.environ.get(name)]
    if missing:
        print(
            "Data not uploaded: required RDS environment settings are not set "
            f"({', '.join(missing)}). Fill them in {env_path} and rerun inference.",
            file=sys.stderr,
        )
        return 0

    if not csv_path.is_file():
        print(f"RDS upload failed: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        import psycopg
    except ImportError:
        print(
            "RDS upload failed: psycopg is not installed for the Python interpreter "
            f"running this script ({sys.executable}). Install it with:\n"
            f'  "{sys.executable}" -m pip install "psycopg[binary]"',
            file=sys.stderr,
        )
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != list(TABLE_COLUMNS):
            print("RDS upload failed: CSV columns do not match pothole_detections schema.", file=sys.stderr)
            return 1
        rows = [tuple(row.get(column) or None for column in TABLE_COLUMNS) for row in reader]

    if not rows:
        print(
            f"Inference completed. No accepted detections were found in {csv_path}; "
            "no data was uploaded to RDS."
        )
        return 0

    try:
        port = int(os.environ.get("RDS_PORT", "5432"))
    except ValueError:
        print("Data not uploaded: RDS_PORT must be a number (usually 5432).", file=sys.stderr)
        return 1
    sslmode = os.environ.get("RDS_SSLMODE", "require")
    insert_sql = """INSERT INTO pothole_detections
        (detection_id, device_id, vehicle_id, detected_at, latitude, longitude,
         length_cm, width_cm, depth_cm, severity, confidence, model_version, image_s3_key)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (detection_id) DO NOTHING"""

    try:
        with psycopg.connect(
            host=os.environ["RDS_HOST"],
            port=port,
            dbname=os.environ["RDS_DATABASE"],
            user=os.environ["RDS_USERNAME"],
            password=os.environ["RDS_PASSWORD"],
            sslmode=sslmode,
            connect_timeout=10,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(insert_sql, rows)
                inserted = cursor.rowcount
    except Exception as exc:
        print(
            f"Data not uploaded to RDS hostname {os.environ['RDS_HOST']}: {exc}",
            file=sys.stderr,
        )
        return 1

    skipped = len(rows) - max(inserted, 0)
    print(
        f"Inference completed. Extracted data uploaded to RDS hostname "
        f"{os.environ['RDS_HOST']}:{port}, database={os.environ['RDS_DATABASE']}; "
        f"CSV rows={len(rows)}, inserted={inserted}, duplicate IDs skipped={skipped}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path", nargs="?", type=Path,
        help="RDS-ready CSV produced by inference (default: newest matching CSV in integration/outputs)",
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH, help=f"Connection settings file (default: {ENV_PATH})")
    args = parser.parse_args()
    csv_path = args.csv_path
    if csv_path is None:
        candidates = sorted(
            (PROJECT_ROOT / "integration" / "outputs").glob("*_potholes_rds.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            parser.error(
                "no *_potholes_rds.csv found in integration/outputs; run inference first "
                "or pass the CSV path explicitly"
            )
        csv_path = candidates[0]
        print(f"Using newest RDS-ready CSV: {csv_path}")
    return upload_csv(csv_path, args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
