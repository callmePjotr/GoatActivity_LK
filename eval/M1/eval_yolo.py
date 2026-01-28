from ultralytics import YOLO
import json
from pathlib import Path


def main():
    # Pfade
    model_path = ""
    data_yaml = "/data.yaml"
    output_dir = Path("")
    output_dir.mkdir(exist_ok=True)

    print("Loading model...")
    model = YOLO(model_path)

    print("\n=== Evaluating on TEST split ===")
    metrics = model.val(
        data=data_yaml,
        split='test',
        batch=8,
        plots=True,
        project=str(output_dir),
        name='yolov8m_test'
    )

    results = {
        'model': 'YOLOv8m-Pose',
        'dataset': 'YOLO_DATASET_4',
        'split': 'test',
        'box_metrics': {
            'mAP@0.5': float(metrics.box.map50),
            'mAP@0.5:0.95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        },
        'pose_metrics': {
            'mAP@0.5': float(metrics.pose.map50),
            'mAP@0.5:0.95': float(metrics.pose.map),
            'precision': float(metrics.pose.mp),
            'recall': float(metrics.pose.mr),
        },
        'speed': metrics.speed,
        'parameters_M': sum(p.numel() for p in model.model.parameters()) / 1e6,
    }

    with open(output_dir / "yolov8m_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
