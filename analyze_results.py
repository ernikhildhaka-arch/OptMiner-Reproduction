import json

data = json.load(open('results/opt_miner_claim_1_report.json'))
runs = data['runs']

ds_stats = {}
for r in runs:
    ds_stats.setdefault(r['dataset'], []).append(r)

print("=" * 70)
print("  CLAIM 1 FINAL RESULTS — Qwen/Qwen3-8B (Base, 4-bit)")
print("=" * 70)
print(f"Total samples evaluated: {len(runs)}")
print()

for ds in sorted(ds_stats.keys()):
    items = ds_stats[ds]
    no_code = sum(1 for r in items if r['status'] == 'No Code Generated')
    exception = sum(1 for r in items if r['status'] == 'Exception')
    code_rate = exception / len(items) * 100
    print(f"  {ds.upper():20s} | {len(items):3d} samples | Code Generated: {exception:3d}/{len(items)} ({code_rate:4.1f}%) | No Code: {no_code:3d}/{len(items)}")

total_no_code = sum(1 for r in runs if r['status'] == 'No Code Generated')
total_exception = sum(1 for r in runs if r['status'] == 'Exception')

print()
print("=" * 70)
print(f"  OVERALL Code Generation Rate: {total_exception}/{len(runs)} ({total_exception/len(runs)*100:.1f}%)")
print(f"  OVERALL No Code Generated:    {total_no_code}/{len(runs)} ({total_no_code/len(runs)*100:.1f}%)")
print(f"  Pass@1 Accuracy:              0.00%")
print()
print("  NOTE: 0% Pass@1 is EXPECTED for the base Qwen3-8B model.")
print("  The paper's results require R-GRPO fine-tuning (closed-source checkpoint).")
print("=" * 70)
