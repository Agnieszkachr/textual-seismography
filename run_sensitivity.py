import os
import pandas as pd
import itertools
import sys
from run_analysis import run_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def main():
    alphas = [1.0, 2.0, 3.0]
    min_sizes = [20, 30, 40]
    penalty_mults = [1.0, 1.43, 2.0]

    grid = list(itertools.product(alphas, min_sizes, penalty_mults))
    print(f"Total Configurations to Evaluate: {len(grid)}\n")

    results_summary = []

    for i, (alpha, min_size, pen) in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Running alpha={alpha}, min_size={min_size}, pen={pen}... ", end="", flush=True)

        res = run_pipeline(
            books=["Isaiah"],
            alpha=alpha,
            min_size=min_size,
            penalty_multiplier=pen,
            verbose=False
        )

        df = res["df"]
        change_points = res["change_points"]
        n_regimes = res["n_regimes"]
        
        target_idx = -1
        verse_to_idx = {v: idx for idx, v in enumerate(df['verse_id'])}
        target_idx = verse_to_idx.get("Isa.40.1", -1)
        
        boundaries = change_points[:-1] if len(change_points) > 0 else []
        all_boundary_positions = str(boundaries)
        
        nearest_boundary_to_ch40 = None
        verse_reference = None
        offset_from_40_1 = None
        within_15_verses = False
        
        if len(boundaries) > 0 and target_idx != -1:
            nearest_boundary_to_ch40 = min(boundaries, key=lambda x: abs(x - target_idx))
            offset_from_40_1 = nearest_boundary_to_ch40 - target_idx
            if nearest_boundary_to_ch40 < len(df):
                verse_reference = df.loc[nearest_boundary_to_ch40, 'verse_id']
            within_15_verses = abs(offset_from_40_1) <= 15
            
        results_summary.append({
            'alpha': alpha,
            'min_size': min_size,
            'penalty': pen,
            'n_regimes': n_regimes,
            'all_boundary_positions': all_boundary_positions,
            'nearest_boundary_to_ch40': nearest_boundary_to_ch40,
            'verse_reference': verse_reference,
            'offset_from_40_1': offset_from_40_1,
            'within_15_verses': within_15_verses
        })
        
        print(f"Done. Regimes: {n_regimes} | 40:1 Hit (±15): {within_15_verses}")

    # Generate CSV
    res_df = pd.DataFrame(results_summary)
    csv_out = os.path.join(OUTPUT_DIR, "sensitivity_grid.csv")
    res_df.to_csv(csv_out, index=False)
    
    # Compute summary stats
    total_perms = len(res_df)
    hits_all = res_df['within_15_verses'].sum()
    pct_all = (hits_all / total_perms) * 100
    
    df_alpha_le_2 = res_df[res_df['alpha'] <= 2.0]
    hits_alpha_le_2 = df_alpha_le_2['within_15_verses'].sum()
    total_alpha_le_2 = len(df_alpha_le_2)
    pct_alpha_le_2 = (hits_alpha_le_2 / total_alpha_le_2) * 100 if total_alpha_le_2 > 0 else 0
    
    min_regimes = res_df['n_regimes'].min()
    max_regimes = res_df['n_regimes'].max()
    
    default_run = res_df[(res_df['alpha'] == 2.0) & (res_df['min_size'] == 30) & (res_df['penalty'] == 1.43)]
    default_regimes = default_run['n_regimes'].values[0] if not default_run.empty else "N/A"
    
    print("\n=== Sensitivity Grid Summary ===")
    print(f"Total permutations: {total_perms}")
    print(f"Boundary within ±15 of ch.40 (all): {hits_all}/{total_perms} ({pct_all:.1f}%)")
    print(f"Boundary within ±15 of ch.40 (alpha<=2.0): {hits_alpha_le_2}/{total_alpha_le_2} ({pct_alpha_le_2:.1f}%)")
    print(f"Regime count range: {min_regimes}-{max_regimes}")
    print(f"Default params (alpha=2.0, min_size=30, pen=1.43): {default_regimes} regimes")

if __name__ == "__main__":
    main()
