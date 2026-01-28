# M1/analyze_by_occlusion_level.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# ============================================================================
# LADE DATEN
# ============================================================================
pred_file = Path(r"L:\Uni\Master\Master Thesis\RE_ID_MODELL\evaluation_thesisM1\results\detailed_predictions_yolov8\predictions.json")
gt_file = Path(r"L:\Uni\Master\Master Thesis\RE_ID_MODELL\evaluation_thesisM1\results\detailed_predictions_yolov8\ground_truth.json")

print("=" * 80)
print("OCCLUSION-STRATIFIED ANALYSIS")
print("=" * 80)

with open(pred_file) as f:
    predictions = json.load(f)

with open(gt_file) as f:
    ground_truths = json.load(f)

# Gruppiere nach Bildern
pred_by_image = {}
for pred in predictions:
    img = pred['image']
    if img not in pred_by_image:
        pred_by_image[img] = []
    pred_by_image[img].append(pred)

gt_by_image = {}
for gt in ground_truths:
    img = gt['image']
    if img not in gt_by_image:
        gt_by_image[img] = []
    gt_by_image[img].append(gt)


# ============================================================================
# MATCHING-FUNKTION
# ============================================================================
def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def match_predictions_to_gt(preds, gts, iou_threshold=0.5):
    """Match predictions to ground truths using Hungarian algorithm"""
    if len(preds) == 0 or len(gts) == 0:
        return [], list(range(len(gts)))

    cost_matrix = np.zeros((len(preds), len(gts)))
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            iou = compute_iou(pred['bbox'], gt['bbox'])
            cost_matrix[i, j] = -iou

    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_gt_indices = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = -cost_matrix[pred_idx, gt_idx]
        if iou >= iou_threshold:
            matches.append((preds[pred_idx], gts[gt_idx], iou))
            matched_gt_indices.add(gt_idx)

    unmatched_gt_indices = [i for i in range(len(gts)) if i not in matched_gt_indices]

    return matches, unmatched_gt_indices


# ============================================================================
# KATEGORISIERE NACH ANZAHL SICHTBARER KEYPOINTS
# ============================================================================
def count_visible_keypoints(gt_keypoints):
    """Count keypoints with visibility > 0.5"""
    return sum(1 for kp in gt_keypoints if kp['visibility'] > 0.5)


# Stratifizierte Metriken
occlusion_levels = {
    '5/5 (vollständig)': {'gt_count': 0, 'matched': 0, 'box_ious': [], 'pose_oks': []},
    '4/5': {'gt_count': 0, 'matched': 0, 'box_ious': [], 'pose_oks': []},
    '3/5 (Minimum)': {'gt_count': 0, 'matched': 0, 'box_ious': [], 'pose_oks': []},
    '<3/5': {'gt_count': 0, 'matched': 0, 'box_ious': [], 'pose_oks': []},
}


def compute_oks(pred_kps, gt_kps, gt_bbox):
    """
    Object Keypoint Similarity (OKS)
    Formula: OKS = Σ exp(-d²/(2*s²*k²)) * δ(v>0) / Σ δ(v>0)

    s = scale (sqrt of bbox area)
    k = keypoint constant (typically 0.05-0.1 for animals)
    d = distance between predicted and ground truth keypoint
    """
    # Berechne scale aus bbox
    bbox_width = gt_bbox[2] - gt_bbox[0]
    bbox_height = gt_bbox[3] - gt_bbox[1]
    scale = np.sqrt(bbox_width * bbox_height)

    # Keypoint constant (angepasst für Tiere)
    kappa = 0.1

    oks_sum = 0
    visible_count = 0

    for pred_kp, gt_kp in zip(pred_kps, gt_kps):
        if gt_kp['visibility'] > 0.5:
            visible_count += 1

            # Euklidische Distanz
            dx = pred_kp['x'] - gt_kp['x']
            dy = pred_kp['y'] - gt_kp['y']
            d = np.sqrt(dx ** 2 + dy ** 2)

            # OKS Beitrag
            oks_sum += np.exp(-d ** 2 / (2 * scale ** 2 * kappa ** 2))

    return oks_sum / visible_count if visible_count > 0 else 0


print(f"\n🔍 Analyzing performance by occlusion level...")

# ============================================================================
# ANALYSIERE JEDES IMAGE
# ============================================================================
for img_name in gt_by_image.keys():
    preds = pred_by_image.get(img_name, [])
    gts = gt_by_image[img_name]

    matches, unmatched_gt_indices = match_predictions_to_gt(preds, gts, iou_threshold=0.5)

    # Analysiere Matches
    for pred, gt, iou in matches:
        # Zähle sichtbare Keypoints
        num_visible = count_visible_keypoints(gt['keypoints'])

        # Kategorisiere
        if num_visible == 5:
            category = '5/5 (vollständig)'
        elif num_visible == 4:
            category = '4/5'
        elif num_visible == 3:
            category = '3/5 (Minimum)'
        else:  # 0, 1, 2
            category = '<3/5'

        # Update Metriken
        occlusion_levels[category]['gt_count'] += 1
        occlusion_levels[category]['matched'] += 1
        occlusion_levels[category]['box_ious'].append(iou)

        # Berechne OKS
        oks = compute_oks(pred['keypoints'], gt['keypoints'], gt['bbox'])
        occlusion_levels[category]['pose_oks'].append(oks)

    # Analysiere Unmatched GTs (Missed Detections)
    for gt_idx in unmatched_gt_indices:
        gt = gts[gt_idx]
        num_visible = count_visible_keypoints(gt['keypoints'])

        if num_visible == 5:
            category = '5/5 (vollständig)'
        elif num_visible == 4:
            category = '4/5'
        elif num_visible == 3:
            category = '3/5 (Minimum)'
        else:
            category = '<3/5'

        occlusion_levels[category]['gt_count'] += 1
        # matched bleibt bei 0 (wurde ja nicht detektiert)

# ============================================================================
# BERECHNE FINALE METRIKEN
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE BY OCCLUSION LEVEL")
print("=" * 80)
print(f"{'Visible Keypoints':<25} {'Anteil':<12} {'Box mAP@0.5':<15} {'Pose mAP@0.5':<15}")
print("-" * 80)

results_table = []

total_gts = sum(level['gt_count'] for level in occlusion_levels.values())

for category in ['5/5 (vollständig)', '4/5', '3/5 (Minimum)', '<3/5']:
    metrics = occlusion_levels[category]

    # Anteil am Testset
    percentage = (metrics['gt_count'] / total_gts * 100) if total_gts > 0 else 0

    # Box mAP@0.5 approximation (IoU > 0.5 als "korrekt")
    if len(metrics['box_ious']) > 0:
        box_map = sum(1 for iou in metrics['box_ious'] if iou > 0.5) / len(metrics['box_ious'])
    else:
        box_map = 0

    # Pose mAP@0.5 approximation (OKS > 0.5 als "korrekt")
    if len(metrics['pose_oks']) > 0:
        pose_map = sum(1 for oks in metrics['pose_oks'] if oks > 0.5) / len(metrics['pose_oks'])
    else:
        pose_map = 0

    # Detection Rate (wie viele GTs wurden überhaupt gematcht?)
    detection_rate = metrics['matched'] / metrics['gt_count'] if metrics['gt_count'] > 0 else 0

    # Korrigiere mAP: Berücksichtige auch missed detections
    # mAP = (True Positives) / (True Positives + False Negatives)
    if metrics['gt_count'] > 0:
        box_map_corrected = (sum(1 for iou in metrics['box_ious'] if iou > 0.5)) / metrics['gt_count']
        pose_map_corrected = (sum(1 for oks in metrics['pose_oks'] if oks > 0.5)) / metrics['gt_count']
    else:
        box_map_corrected = 0
        pose_map_corrected = 0

    results_table.append({
        'category': category,
        'percentage': percentage,
        'box_map': box_map_corrected,
        'pose_map': pose_map_corrected,
        'detection_rate': detection_rate,
        'count': metrics['gt_count']
    })

    print(f"{category:<25} {percentage:>5.0f}%      {box_map_corrected:>8.3f}        {pose_map_corrected:>8.3f}")

# ============================================================================
# DETAILLIERTE STATISTIK
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)
print(f"{'Category':<25} {'Count':<10} {'Detected':<12} {'Detection Rate'}")
print("-" * 80)

for result in results_table:
    detected = int(result['detection_rate'] * result['count'])
    print(f"{result['category']:<25} {result['count']:<10} {detected:<12} {result['detection_rate']:.1%}")

# ============================================================================
# VISUALISIERUNG
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

categories = ['5/5\n(vollständig)', '4/5', '3/5\n(Minimum)', '<3/5']
percentages = [r['percentage'] for r in results_table]
box_maps = [r['box_map'] for r in results_table]
pose_maps = [r['pose_map'] for r in results_table]
detection_rates = [r['detection_rate'] for r in results_table]

# Plot 1: Performance by Occlusion
x = np.arange(len(categories))
width = 0.25

bars1 = axes[0].bar(x - width, box_maps, width, label='Box mAP@0.5', color='#3498db')
bars2 = axes[0].bar(x, pose_maps, width, label='Pose mAP@0.5', color='#2ecc71')
bars3 = axes[0].bar(x + width, detection_rates, width, label='Detection Rate', color='#e74c3c')

axes[0].set_ylabel('Score', fontsize=11)
axes[0].set_xlabel('Visible Keypoints', fontsize=11)
axes[0].set_title('Performance by Occlusion Level (YOLOv11m-Pose)', fontsize=13, weight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0.75, 1.05])

# Annotiere Werte
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Dataset Distribution
colors_dist = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
bars = axes[1].bar(categories, percentages, color=colors_dist, edgecolor='black', linewidth=1.5)

axes[1].set_ylabel('Percentage of Test Set', fontsize=11)
axes[1].set_xlabel('Visible Keypoints', fontsize=11)
axes[1].set_title('Distribution of Occlusion Levels', fontsize=13, weight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Annotiere Counts und Percentages
for bar, pct, result in zip(bars, percentages, results_table):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{pct:.0f}%\n(n={result["count"]})',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('M1/results/occlusion_stratified_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: M1/results/occlusion_stratified_analysis.png")

# ============================================================================
# SPEICHERE ERGEBNISSE
# ============================================================================
with open('M1/results/occlusion_stratified_results.json', 'w') as f:
    json.dump(results_table, f, indent=2)
print("✅ Saved: M1/results/occlusion_stratified_results.json")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Drucke Markdown-Tabelle für Thesis
print("\n📋 MARKDOWN TABLE FOR THESIS:")
print("```")
print("| Sichtbare Keypoints | Anteil Testset | Box mAP@.5 | Pose mAP@.5 |")
print("|---------------------|----------------|------------|-------------|")
for result in results_table:
    print(f"| {result['category']:<19} | {result['percentage']:>5.0f}%          | "
          f"{result['box_map']:>8.3f}   | {result['pose_map']:>8.3f}    |")
print("```")