from pathlib import Path
from typing import Dict, Any

import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_experiment_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create the folders needed by the debug RunPod reproduction.

    Compared with the notebook, checkpoints and heatmaps are namespaced by
    experiment_id so repeated runs do not overwrite each other.
    """
    project_root = Path(config["project_root"])
    output_dir = Path(config["output_dir"])
    experiment_id = config["experiment_id"]

    data_dir = project_root / "data"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    checkpoint_dir = output_dir / "checkpoints" / experiment_id
    metrics_dir = output_dir / "metrics"
    heatmap_dir = output_dir / "heatmaps" / experiment_id
    figure_dir = output_dir / "figures"
    log_dir = output_dir / "logs"

    dirs = {
        "project_root": project_root,
        "output_dir": output_dir,
        "data_dir": data_dir,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "checkpoint_dir": checkpoint_dir,
        "metrics_dir": metrics_dir,
        "heatmap_dir": heatmap_dir,
        "figure_dir": figure_dir,
        "log_dir": log_dir,
    }

    for path in dirs.values():
        ensure_dir(path)

    return dirs


def save_config_snapshot(config: Dict[str, Any], dirs: Dict[str, Path]) -> Path:
    """Save the exact config used by a run."""
    path = dirs["checkpoint_dir"] / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def print_output_check(dirs: Dict[str, Path]) -> None:
    """Equivalent to the notebook's final output check."""
    print("Project root:", dirs["project_root"])

    print("\nMetrics files:")
    for path in sorted(dirs["metrics_dir"].glob("*.csv")):
        print("-", path)

    print("\nCheckpoint files:")
    for path in sorted(dirs["checkpoint_dir"].glob("*.pt")):
        print("-", path)

    print("\nHeatmap folders:")
    for path in sorted(dirs["heatmap_dir"].glob("*")):
        print("-", path)

    print("\nInterim files:")
    for path in sorted(dirs["interim_dir"].glob("*.csv")):
        print("-", path)

def zip_outputs(output_dir: Path, project_root: Path, zip_name: str = "prototype_outputs") -> Path:
    """Optional equivalent of the notebook's final zip/export step."""
    import shutil

    zip_base = project_root / zip_name
    zip_path = shutil.make_archive(
        base_name=str(zip_base),
        format="zip",
        root_dir=str(output_dir),
    )

    print("Created zip file:", zip_path)
    return Path(zip_path)