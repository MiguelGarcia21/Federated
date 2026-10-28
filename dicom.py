
from pathlib import Path
import pandas as pd
import pydicom

# Adjust these paths if necessary
from pathlib import Path
import pandas as pd
import pydicom

# Project root: directory containing this Python script
PROJECT_ROOT = Path(__file__).resolve().parent

XAI_CSV = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "exp01_centralized_vindr_binary_seed42_xai_image_ids.csv"
)

DICOM_DIR = PROJECT_ROOT / "data" / "raw" / "vindr" / "train"

OUTPUT_CSV = PROJECT_ROOT / "vindr_xai59_dicom_dimensions.csv"

# Load the existing 200-image XAI subset
xai = pd.read_csv(XAI_CSV)

# Select only abnormal images
abnormal = xai[xai["binary_label"] == 1].copy()

print(f"Abnormal images: {len(abnormal)}")

records = []

for image_id in abnormal["image_id"]:
    dicom_path = DICOM_DIR / f"{image_id}.dicom"

    if not dicom_path.exists():
        print(f"Missing: {dicom_path}")
        records.append({
            "image_id": image_id,
            "height": None,
            "width": None,
            "status": "missing"
        })
        continue

    try:
        # Read header only, without pixel data
        ds = pydicom.dcmread(
            str(dicom_path),
            stop_before_pixels=True
        )

        records.append({
            "image_id": image_id,
            "height": int(ds.Rows),
            "width": int(ds.Columns),
            "status": "ok"
        })

    except Exception as e:
        print(f"Error reading {image_id}: {e}")
        records.append({
            "image_id": image_id,
            "height": None,
            "width": None,
            "status": "error"
        })

# Export results
results = pd.DataFrame(records)
results.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Successful: {(results['status'] == 'ok').sum()}")
print(f"Missing/errors: {(results['status'] != 'ok').sum()}")
