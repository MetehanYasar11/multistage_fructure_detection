import json

# Load results
with open('runs/full_pipeline_validation/pipeline_results.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("DETAILED ANALYSIS OF PIPELINE RESULTS")
print("=" * 80)
print()

# False Negatives (Missed fractured images)
fn_results = [r for r in data['detailed_results'] if r['image_level_result'] == 'FN']
print(f"FALSE NEGATIVES (Missed Fractured Images): {len(fn_results)}")
for r in fn_results:
    print(f"  ❌ {r['class']}/{r['image_name']}")
    print(f"      Stage 1: {r['num_rcts_detected']} RCTs detected")
    print(f"      Stage 2: {r['num_fractures_predicted']} fractures predicted (should be ≥1)")
    print()

# False Positives (Healthy images flagged as fractured)
fp_results = [r for r in data['detailed_results'] if r['image_level_result'] == 'FP']
print(f"\nFALSE POSITIVES (Healthy Images Flagged): {len(fp_results)}")
for i, r in enumerate(fp_results[:5]):  # Show first 5
    print(f"  ⚠️  {r['class']}/{r['image_name']}")
    print(f"      Stage 1: {r['num_rcts_detected']} RCTs detected")
    print(f"      Stage 2: {r['num_fractures_predicted']} fractures predicted")
if len(fp_results) > 5:
    print(f"  ... and {len(fp_results) - 5} more")
print()

# True Positives (Correctly detected fractured images)
tp_results = [r for r in data['detailed_results'] if r['image_level_result'] == 'TP']
print(f"TRUE POSITIVES (Correctly Detected): {len(tp_results)}")
print(f"  ✅ {len(tp_results)}/60 fractured images correctly identified")
print()

# Sample TP with most fractures predicted
tp_sorted = sorted(tp_results, key=lambda x: x['num_fractures_predicted'], reverse=True)
print("  Top 5 with most fracture predictions:")
for i, r in enumerate(tp_sorted[:5]):
    print(f"    {i+1}. {r['image_name']}: {r['num_fractures_predicted']} fractures found in {r['num_rcts_detected']} RCTs")
print()

# Stage statistics
print("STAGE STATISTICS:")
print(f"  Stage 1 (RCT Detection):")
print(f"    - Total RCT crops: {data['stage1_metrics']['total_rcts_detected']}")
print(f"    - Images with RCT: {data['stage1_metrics']['images_with_rct']}/73")
print()
print(f"  Stage 2 (Fracture Classification):")
print(f"    - Total fracture predictions: {data['stage2_metrics']['total_fractures_predicted']}")
print(f"    - Images with fracture: {data['stage2_metrics']['images_with_fracture_predicted']}/73")
print()

# Performance metrics
metrics = data['classification_metrics']
print("FINAL METRICS:")
print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
print(f"  Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
print(f"  F1 Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
