import pandas as pd
import numpy as np
import argparse
import sys

def evaluate(csv_path, tolerances=[5, 10, 15, 20]):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        sys.exit(1)

    verse_to_idx = {v: i for i, v in enumerate(df['verse_id'])}
    
    # 1. Classical Multi-Authorial Macro-Seams
    consensus_seams = ["Isa.24.1", "Isa.36.1", "Isa.40.1", "Isa.56.1"]
    
    # 2. Expanded Structural Seams (including Major Thematic/Genre Shifts)
    # Includes Messianic Oracles (11:1, 4:2) and major thematic transitions recorded in commentaries.
    expanded_seams = consensus_seams + ["Isa.4.2", "Isa.8.23", "Isa.11.1"]
    
    gt_strict_indices = []
    for seam in consensus_seams:
        if seam in verse_to_idx:
            gt_strict_indices.append(verse_to_idx[seam])
            
    gt_expanded_indices = []
    for seam in expanded_seams:
        if seam in verse_to_idx:
            gt_expanded_indices.append(verse_to_idx[seam])

    pred_indices = df[df['Is_Regime_Change'] == True].index.tolist()

    print("============================================================")
    print(f"  QUANTITATIVE EVALUATION: {csv_path}")
    print(f"  Total Predictions: {len(pred_indices)}")
    print(f"  Strict Multi-Authorial Ground Truths: {len(gt_strict_indices)}")
    print(f"  Expanded Structural Ground Truths: {len(gt_expanded_indices)}")
    print("============================================================")

    def calculate_metrics(gt_indices):
        results = {}
        for tol in tolerances:
            tp = 0
            matched_gt = set()
            for p_idx in pred_indices:
                for g_idx in gt_indices:
                    if g_idx in matched_gt: continue
                    if abs(p_idx - g_idx) <= tol:
                        tp += 1
                        matched_gt.add(g_idx)
                        break
            
            fp = len(pred_indices) - tp
            fn = len(gt_indices) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            results[tol] = {'tp': tp, 'fp': fp, 'fn': fn, 'prec': precision, 'rec': recall}
        return results

    strict_results = calculate_metrics(gt_strict_indices)
    expanded_results = calculate_metrics(gt_expanded_indices)
    
    print("\n--- PERFORMANCE VS. STRICT EDITORIAL SEAMS ---")
    print(f"{'Tol':<5} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    for tol in sorted(tolerances):
        r = strict_results[tol]
        print(f"±{tol:<4} | {r['tp']:<4} | {r['fp']:<4} | {r['fn']:<4} | {r['prec']:<10.2f} | {r['rec']:<10.2f}")

    print("\n--- PERFORMANCE VS. EXPANDED THEMATIC/STRUCTURAL SEAMS ---")
    print(f"{'Tol':<5} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    for tol in sorted(tolerances):
        r = expanded_results[tol]
        print(f"±{tol:<4} | {r['tp']:<4} | {r['fp']:<4} | {r['fn']:<4} | {r['prec']:<10.2f} | {r['rec']:<10.2f}")
    
    # Return the strict metrics for backwards compatibility with the sensitivity script
    return strict_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate predicted structural boundaries against consensus.")
    parser.add_argument("--csv", required=True, help="Path to the seismograph output CSV")
    parser.add_argument("--tolerances", type=int, nargs='+', default=[5, 10, 15, 20], help="Tolerance windows to evaluate")
    args = parser.parse_args()

    evaluate(args.csv, args.tolerances)
