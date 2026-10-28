from pathlib import Path
import pandas as pd
import pydicom

# CSV is inside your project directory
XAI_CSV = Path(
    "/workspace/thesis-fl-xai/data/interim/"
    "exp01_centralized_vindr_binary_seed42_xai_image_ids.csv"
)

# DICOM data is outside your project directory
DICOM_DIR = Path("/workspace/data/raw/vindr/train")

# Save results inside your project directory
OUTPUT_CSV = Path(
    "/workspace/thesis-fl-xai/data/interim/vindr_xai59_dicom_dimensions.csv"
)

xai = pd.read_csv(XAI_CSV)

abnormal = xai[xai["binary_label"] == 1].copy()

print(f"Abnormal images: {len(abnormal)}")
print(f"DICOM directory exists: {DICOM_DIR.exists()}")
print(f"CSV exists: {XAI_CSV.exists()}")

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
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)

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

results = pd.DataFrame(records)
results.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Successful: {(results['status'] == 'ok').sum()}")
print(f"Missing/errors: {(results['status'] != 'ok').sum()}")