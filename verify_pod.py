import importlib
import sys

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("pydicom", "pydicom"),
    ("flwr", "flwr"),
    ("ray", "ray"),
    ("yaml", "pyyaml"),
    ("cv2", "opencv-python"),
]

def check_import(module_name: str, package_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        print(f"[OK] {package_name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {package_name}: {exc}")
        return False

def main() -> int:
    print("Checking Python environment...")
    print("Python:", sys.version)

    all_ok = True

    for module_name, package_name in REQUIRED_PACKAGES:
        all_ok = check_import(module_name, package_name) and all_ok

    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("GPU: CPU only / CUDA not available")
    except Exception as exc:
        print("[FAIL] CUDA/Torch check:", exc)
        all_ok = False

    if all_ok:
        print("All required imports successful.")
        return 0

    print("Some imports failed. Check requirements.txt or the active environment.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())