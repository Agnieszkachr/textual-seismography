import numpy as np
import pandas as pd
from run_analysis import run_pipeline

def diagnostic_1():
    print("--- DIAGNOSTIC 1: Leviticus Boundaries at pen=1.43 ---")
    res = run_pipeline(books=["Leviticus"], alpha=2.0, min_size=30, penalty_multiplier=1.43, verbose=False)
    print(f"Total regime shifts: {res['n_regimes']}")
    for verse in res["change_point_verses"]:
        print(f"  Boundary at: {verse}")
    print()
    return res["df"]

def diagnostic_4(lev_df, isa_df):
    print("--- DIAGNOSTIC 4: CFI Distribution Comparison ---")
    isa_cfi = isa_df['CFI_mag'].values
    lev_cfi = lev_df['CFI_mag'].values
    
    # Use ddof=1 for sample std to match typical expectations, or 0 if population.
    # The user asked for np.std() which defaults to ddof=0, let's use default.
    print(f"Isaiah    CFI: mean={np.mean(isa_cfi):.3f}, std={np.std(isa_cfi):.3f}, max={np.max(isa_cfi):.3f}, median={np.median(isa_cfi):.3f}")
    print(f"Leviticus CFI: mean={np.mean(lev_cfi):.3f}, std={np.std(lev_cfi):.3f}, max={np.max(lev_cfi):.3f}, median={np.median(lev_cfi):.3f}")
    print()

def diagnostic_3():
    print("--- DIAGNOSTIC 3: Control Books at pen=1.43 ---")
    for book in ["Ruth", "Jonah", "Habakkuk"]:
        res = run_pipeline(books=[book], alpha=2.0, min_size=30, penalty_multiplier=1.43, verbose=False)
        print(f"{book}: {res['n_regimes']} regime shifts")
    print()

def diagnostic_2():
    print("--- DIAGNOSTIC 2: Penalty Sweep ---")
    penalties = [1.0, 1.43, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for pen in penalties:
        lev_res = run_pipeline(books=["Leviticus"], alpha=2.0, min_size=30, penalty_multiplier=pen, verbose=False)
        isa_res = run_pipeline(books=["Isaiah"], alpha=2.0, min_size=30, penalty_multiplier=pen, verbose=False)
        
        lev_shifts = lev_res['n_regimes']
        isa_shifts = isa_res['n_regimes']
        
        isa_df = isa_res['df']
        # Handle cases where verse might not exactly match
        idx_match = isa_df.index[isa_df['verse_id'] == 'Isa.40.1'].tolist()
        idx_40_1 = idx_match[0] if idx_match else 765
        
        has_pd = "NO"
        for cp in isa_res['change_points'][:-1]:
            if abs(cp - idx_40_1) <= 15:
                has_pd = "YES"
                break
                
        print(f"pen={pen:5.2f}: Leviticus={lev_shifts:2d} shifts, Isaiah={isa_shifts:2d} shifts, Proto/Deutero present={has_pd}")
    print()

if __name__ == '__main__':
    lev_df = diagnostic_1()
    
    # Get Isaiah at default for CFI
    isa_res = run_pipeline(books=["Isaiah"], alpha=2.0, min_size=30, penalty_multiplier=1.43, verbose=False)
    
    diagnostic_4(lev_df, isa_res['df'])
    diagnostic_3()
    diagnostic_2()
