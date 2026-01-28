from mmengine.config import Config
from mmengine.runner import Runner
import json
import os

CFG = "configs/custom/faster_rcnn_r50_custom.py"
CKPT = "work_dirs/fasterrcnn_animal/best_coco_bbox_mAP_epoch_21.pth"

cfg = Config.fromfile(CFG)
cfg.load_from = CKPT
cfg.work_dir = "work_dirs/fasterrcnn_animal"  

runner = Runner.from_cfg(cfg)
metrics = runner.test()


precision = coco["precision"]   # [T, R, K, A, M]
recall    = coco["recall"]      # [T, K, A, M]

IOU_05 = 0
CLS    = 0
AREA   = 0
MAXDET = 2  

prec_vals = precision[IOU_05, :, CLS, AREA, MAXDET]
prec_vals = prec_vals[prec_vals > -1]

precision_05 = float(prec_vals.mean())
recall_05    = float(recall[IOU_05, CLS, AREA, MAXDET])
f1_05        = 2 * precision_05 * recall_05 / (precision_05 + recall_05 + 1e-9)

metrics = runner.test()

result = {
    "model": "Faster R-CNN R50 FPN",
    "dataset": "YOLO_DATASET_4",
    "split": "test",
    "box_metrics": {
        "mAP@0.5": float(metrics["coco/bbox_mAP_50"]),
        "mAP@0.5:0.95": float(metrics["coco/bbox_mAP"]),
    }
}


with open("fasterrcnn_metrics.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
