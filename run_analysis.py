"""
run_strict_analysis.py - Revised Rigorous Statistical Re-Analysis of Isaiah

Implements:
1. Global Z-score standardisation (instead of rolling N=5) to break the mathematical Z-score ceiling.
2. Continuous Lexical Discounting: Raw perplexity scores are mathematically discounted based on the ratio of rare vocabulary (hapax legomena) in the verse, preventing OOV artefacts from artificially inflating the structural signal.
3. FDR Correction (Benjamini-Hochberg) on the global Z-scores across all 1,291 verses.
4. Composite Vector Magnitude CFI with strict FDR significance testing.
"""
import os
import json
import argparse
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, pearsonr, spearmanr, percentileofscore
from statsmodels.stats.multitest import fdrcorrection
import ruptures as rpt
from report_builder import build_report_html
from data_manager import DataManager


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

def build_lexicon():
    from data_manager import DataManager
    dm = DataManager()
    freq = {}
    cached_files = [f for f in os.listdir(SRC_DIR) if f.endswith('.xml')]
    print(f"[Lexicon] Building frequency lexicon from {len(cached_files)} cached biblical books...")
    for file in cached_files:
        book_name = file.replace('.xml', '')
        try:
            xml_str = dm.get_text(book_name)
            verses = dm.parse_verses(xml_str)
            for v in verses:
                for w in dm.clean_hebrew(v["text"]).split():
                    freq[w] = freq.get(w, 0) + 1
        except Exception as e:
            pass
    return freq

def parse_book_arg(arg):
    match = re.match(r"(.*?)(?:\s+(\d+)(?:-(\d+))?)?$", arg.strip())
    book_name = match.group(1).strip()
    start_chap = int(match.group(2)) if match.group(2) else None
    end_chap = int(match.group(3)) if match.group(3) else start_chap
    return book_name, start_chap, end_chap

def load_or_compute_scores_for_selections(book_selections, args):
    """
    Given a list of (book_url, start_chapter, end_chapter) tuples,
    fetch the verses, compute (or load from cache) the raw perplexity scores,
    and then combine, globally standardise, dynamically discount, 
    and run change-point detection using the specified hyper-parameters.
    """
    all_df_gpt = []
    all_df_dicta = []
    
    dm = DataManager() # Initialize DataManager here

    for book_arg in book_selections:
        book_name, start_chap, end_chap = parse_book_arg(book_arg)
        
        # Unique cache filename for partial subsets to prevent overriding the full book
        if start_chap is not None:
             subset_label = f"_{start_chap}to{end_chap}"
        else:
             subset_label = ""
             
        safe_book = book_name.lower().replace(" ", "_") + subset_label
        gpt_path = os.path.join(OUTPUT_DIR, f"{safe_book}_gpt.scores.json")
        dicta_path = os.path.join(OUTPUT_DIR, f"{safe_book}_dicta.scores.json")
        
        if not os.path.isfile(gpt_path) or not os.path.isfile(dicta_path):
            print(f"[Inference] Cache missing for '{book_arg}'. Computing neural perplexities...")
            from seismograph_engine import SeismographEngine
            from dicta_engine import DictaEngine
            
            xml_str = dm.get_text(book_name)
            all_verses = dm.parse_verses(xml_str)
            
            # Pre-inference chapter filtering
            if start_chap is not None:
                verses = []
                for v in all_verses:
                    chap_num = int(v["verse_id"].split('.')[1])
                    if start_chap <= chap_num <= end_chap:
                        verses.append(v)
            else:
                verses = all_verses
                
            if not verses:
                print(f"[Warning] No verses found for selection '{book_arg}'. Skipping.")
                continue
                
            text_units = [dm.clean_hebrew(v["text"]) for v in verses]
            
            # 1. GPT-Neo
            gpt_engine = SeismographEngine()
            gpt_records = []
            for i, v in enumerate(verses):
                ppl = gpt_engine._score_transition(text_units, i)
                gpt_records.append({"verse_id": v["verse_id"], "text_snippet": text_units[i], "perplexity_score": ppl})
                if (i + 1) % 10 == 0 or (i + 1) == len(verses):
                    sys.stdout.write(f"\r  GPT-Neo: {i+1}/{len(verses)} verses")
                    sys.stdout.flush()
            print()
            with open(gpt_path, "w", encoding="utf-8") as f:
                json.dump({"book": book_name, "model": "gpt_neo", "records": gpt_records}, f, indent=2, ensure_ascii=False)
                
            # 2. DictaBERT
            dicta_engine = DictaEngine()
            dicta_records = []
            for i, v in enumerate(verses):
                ppl = dicta_engine._compute_pll(text_units[i])
                dicta_records.append({"verse_id": v["verse_id"], "text_snippet": text_units[i], "perplexity_score": ppl})
                if (i + 1) % 10 == 0 or (i + 1) == len(verses):
                    sys.stdout.write(f"\r  DictaBERT: {i+1}/{len(verses)} verses")
                    sys.stdout.flush()
            print()
            with open(dicta_path, "w", encoding="utf-8") as f:
                json.dump({"book": book_name, "model": "dictabert", "records": dicta_records}, f, indent=2, ensure_ascii=False)
                
            print(f"[Inference] Computed and cached {len(verses)} verses for {book_arg}.")

        with open(gpt_path, "r", encoding="utf-8") as f:
            b_df_gpt = pd.DataFrame(json.load(f)["records"])
        with open(dicta_path, "r", encoding="utf-8") as f:
            b_df_dicta = pd.DataFrame(json.load(f)["records"])
            
        all_df_gpt.append(b_df_gpt)
        all_df_dicta.append(b_df_dicta)

    if not all_df_gpt:
        raise ValueError("No valid books or data found to sequence.")
        
    df_gpt = pd.concat(all_df_gpt, ignore_index=True)
    df_dicta = pd.concat(all_df_dicta, ignore_index=True)
    return df_gpt, df_dicta

def compute_rare_ratio(text, lexicon, rare_threshold=2):
    """Calculate the ratio of rare vocabulary in a verse.
    
    Used to dynamically discount perplexity scores to control for out-of-vocabulary anomalies,
    as described in Section 2: Method (Continuous Lexical Discounting).
    """
    words = str(text).split()
    if not words:
        return 0.0
    rare_count = sum(1 for w in words if lexicon.get(w, 0) <= rare_threshold)
    return rare_count / len(words)

def run_pipeline(books, alpha=2.0, min_size=30, penalty_multiplier=1.43, limit=None, verbose=True):
    report_title = " + ".join(books)
    if limit:
        report_title += f" (first {limit} verses)"

    if verbose:
        print("============================================================")
        print(f"  NEURAL SEISMOGRAPHY: ENSEMBLE PIPELINE ({report_title})")
        print("============================================================")

    lexicon = build_lexicon()
    # Create a dummy args object for load_or_compute_scores_for_selections since it uses args.book
    class DummyArgs: pass
    dummy_args = DummyArgs()
    dummy_args.book = books
    
    df_gpt, df_dicta = load_or_compute_scores_for_selections(books, dummy_args)
    
    if limit:
        df_gpt = df_gpt.head(limit)
        df_dicta = df_dicta.head(limit)

    df = pd.DataFrame({
        "verse_id": df_gpt["verse_id"],
        "text": df_gpt["text_snippet"],
        "raw_ppl_gpt": df_gpt["perplexity_score"],
        "raw_ppl_dicta": df_dicta["perplexity_score"]
    })

    # 3. Compute Lexical Frequency Weight (Rare Word Ratio)
    # Rare words artificially inflate perplexity. We discount the perplexity linearly
    # relative to the concentration of rare words in the verse.
    df["rare_ratio"] = df["text"].apply(lambda t: compute_rare_ratio(t, lexicon, rare_threshold=2))
    
    df["disc_ppl_gpt"] = df["raw_ppl_gpt"] / (1.0 + alpha * df["rare_ratio"])
    df["disc_ppl_dicta"] = df["raw_ppl_dicta"] / (1.0 + alpha * df["rare_ratio"])
    
    if verbose:
        print(f"\n[Lexical Discounting] Applied discount up to factor {1+alpha} for verses with high hapax counts.")
    
    def global_z(series):
        """Standardise a series globally. Implements Section 2: Method (Global Standardisation & Point Anomalies)."""
        return (series - series.mean()) / series.std(ddof=1)
        
    df["global_z_gpt"] = global_z(df["disc_ppl_gpt"])
    df["global_z_dicta"] = global_z(df["disc_ppl_dicta"])
    
    if verbose:
        print(f"[Global Z-Scores] Max GPT Z: {df['global_z_gpt'].max():.2f} | Max Dicta Z: {df['global_z_dicta'].max():.2f}")
    
    # 5. Formalise 'Shared Seams' via Vector Magnitude CFI
    # The Z-scores are merged via Euclidean magnitude into a Continuous Fracture Index.
    # While treating the two models as independent dimensions is a geometric simplifying assumption,
    # it provides an intuitive heuristic for dual-model divergence in the pilot study.
    # Implements Section 2: Method (Continuous Fracture Index).
    z_gpt_pos = np.maximum(df["global_z_gpt"], 0)
    z_dicta_pos = np.maximum(df["global_z_dicta"], 0)
    df["CFI_mag"] = np.sqrt(z_gpt_pos**2 + z_dicta_pos**2)
    
    # 6. P-Values and FDR Correction
    # Raw independent p-values
    df["p_val_gpt"] = norm.sf(df["global_z_gpt"])
    df["p_val_dicta"] = norm.sf(df["global_z_dicta"])
    
    # FDR correction via Benjamini-Hochberg for the two independent models
    rej_gpt, p_adj_gpt = fdrcorrection(df["p_val_gpt"], alpha=0.05)
    rej_dicta, p_adj_dicta = fdrcorrection(df["p_val_dicta"], alpha=0.05)
    df["p_adj_gpt"] = p_adj_gpt
    df["p_adj_dicta"] = p_adj_dicta
    
    # Strict Shared Seam: Significant in BOTH models post-FDR
    df["Strict_Shared_Seam"] = rej_gpt & rej_dicta
    
    # CFI Magnitude testing: The squared magnitude of two positive clamped standard normals
    # roughly follows a mix of 0 and ChiSquare(df=2). For the right tail (large CFI), 
    # we can conservatively use ChiSquare(df=2) for the composite signal.
    df["p_val_cfi"] = chi2.sf(df["CFI_mag"]**2, df=2)
    rej_cfi, p_adj_cfi = fdrcorrection(df["p_val_cfi"], alpha=0.05)
    df["p_adj_cfi"] = p_adj_cfi
    df["Significant_CFI_Seam"] = rej_cfi
    
    # 6.5 Inter-Model Correlation and Percentiles
    pearson_r, pearson_p = pearsonr(df["global_z_gpt"], df["global_z_dicta"])
    spearman_rho, spearman_p = spearmanr(df["global_z_gpt"], df["global_z_dicta"])
    
    if verbose:
        print(f"\n[Inter-Model Correlation]")
        print(f"  Pearson r:  {pearson_r:.4f}  (p = {pearson_p:.3e})")
        print(f"  Spearman rho: {spearman_rho:.4f}  (p = {spearman_p:.3e})")
    
    cfi_values = df["CFI_mag"].values
    df["cfi_percentile"] = df["CFI_mag"].apply(
        lambda x: round(percentileofscore(cfi_values, x, kind='rank'), 2)
    )
    
    if verbose:
        print("\n[Benjamini-Hochberg FDR @ alpha=0.05]")
        print(f"  GPT-Neo global significant seams:   {rej_gpt.sum()}")
        print(f"  DictaBERT global significant seams: {rej_dicta.sum()}")
        print(f"  Strict Shared Seams (Both Models):  {df['Strict_Shared_Seam'].sum()}")
        print(f"  Composite CFI significant seams:    {rej_cfi.sum()}")
        
        print("\n============================================================")
        print("  TOP 15 REDACTIONAL SEAMS BY CUMULATIVE SIGNIFICANCE")
        print("============================================================")
        top_seams = df.sort_values('CFI_mag', ascending=False).head(15)
        for _, r in top_seams.iterrows():
            sig_marker = "***" if r.Significant_CFI_Seam else ""
            dual_marker = "++" if r.Strict_Shared_Seam else ("+" if (r.p_adj_gpt<0.05 or r.p_adj_dicta<0.05) else "")
            print(f"  {r.verse_id:<10} CFI={r.CFI_mag:.3f}  p_CFI={r.p_adj_cfi:.3e}  p_GPT={r.p_adj_gpt:.3e}  p_Dict={r.p_adj_dicta:.3e} {sig_marker}{dual_marker}")

        df['chapter'] = df['verse_id'].apply(lambda x: int(x.split('.')[1]))
        ch_mean = df.groupby('chapter')[['CFI_mag']].mean()
        print("\n============================================================")
        print("  TOP 10 ANOMALOUS CHAPTERS (GLOBAL CFI MAGNITUDE)")
        print("============================================================")
        print(ch_mean.sort_values('CFI_mag', ascending=False).head(10))

    if verbose:
        print("\n============================================================")
        print("  PELT CHANGE-POINT DETECTION (MACRO REGIMES)")
        print("============================================================")
    # The standard L2 cost function is rejected because it overfits to mean-variance spikes in chaotic
    # local regions. To detect sustained baseline shifts in the underlying LM probability distribution,
    # we use the RBF (Radial Basis Function) kernel given its specific sensitivity to non-parametric,
    # underlying distributional shifts. This is combined with min_size to strictly enforce macro-strata.
    signal = df["CFI_mag"].values.reshape(-1, 1)
    
    df.reset_index(drop=True, inplace=True)
    df['Regime_ID'] = 0
    df['Is_Regime_Change'] = False
    
    change_points = []
    change_point_verses = []
    
    if len(signal) > min_size:
        # Default jump = 5 based on diagnostic tests, ensuring consistency
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=5).fit(signal)
        change_points = algo.predict(pen=penalty_multiplier)
        
        if verbose:
            print(f"  Detected {len(change_points)-1} regime changes (macro redactional boundaries)")
        
        current_regime = 1
        start_idx = 0
        for cp in change_points:
            df.loc[start_idx:cp-1, 'Regime_ID'] = current_regime
            if start_idx > 0:
                v_id = df.loc[start_idx, 'verse_id']
                change_point_verses.append(v_id)
                if verbose:
                    print(f"  -> Regime {current_regime} starts abruptly at: {v_id}")
                df.loc[start_idx, 'Is_Regime_Change'] = True
            start_idx = cp
            current_regime += 1
    else:
        if verbose:
            print(f"  [Sequence too short] Detected 0 regime changes. Sequence length ({len(signal)}) <= min_size ({min_size}).")
        df['Regime_ID'] = 1

    return {
        "df": df,
        "change_points": change_points,
        "change_point_verses": change_point_verses,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "n_regimes": max(0, len(change_points) - 1)
    }

def main():
    parser = argparse.ArgumentParser(description="Ensemble Seismograph for Biblical Redaction Detection")
    parser.add_argument("--book", nargs='+', default=["Isaiah"], 
                        help="Book names to analyze. You can specify subsets, e.g., 'Genesis 1-3'. Multiple books are concatenated in order.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first LIMIT verses in the concatenated sequence.")
    
    # DHQ Sensitivity Analysis Parameters
    parser.add_argument("--alpha", type=float, default=2.0, help="Lexical discounting coefficient (default 2.0)")
    parser.add_argument("--min-size", type=int, default=30, help="PELT minimum segment length in verses (default 30)")
    parser.add_argument("--penalty-multiplier", type=float, default=1.43, help="PELT cost penalty multiplier (default 1.43)")
    parser.add_argument("--controls", action="store_true", help="Run the pipeline on a predefined set of control books (no output html, just stats).")
    
    args = parser.parse_args()

    if args.controls:
        control_books = [
            "Ruth",        # 85 verses
            "Jonah",       # 48 verses
            "Habakkuk",    # 56 verses
            "Leviticus",   # 859 verses
            "Deuteronomy", # 959 verses
        ]
        print("============================================================")
        print("  CONTROL BATCH RUNNER")
        print("============================================================")
        for cbook in control_books:
            print(f"Running Control: {cbook}...")
            try:
                res = run_pipeline(books=[cbook], alpha=args.alpha, min_size=args.min_size, 
                                   penalty_multiplier=args.penalty_multiplier, verbose=False)
                verses_length = len(res["df"])
                print(f"  -> {cbook} ({verses_length} verses): {res['n_regimes']} regime shifts detected")
            except Exception as e:
                print(f"  -> {cbook}: Error occurred ({e})")
        return

    # Normal execution via run_pipeline
    res = run_pipeline(
        books=args.book,
        alpha=args.alpha,
        min_size=args.min_size,
        penalty_multiplier=args.penalty_multiplier,
        limit=args.limit,
        verbose=True
    )
    
    df = res["df"]

    # Save to CSV
    safe_title = "_".join([re.sub(r'[^a-z0-9_]', '', s.lower().replace(" ", "_").replace("-", "to")) for s in args.book])[:80]
    csv_out = os.path.join(OUTPUT_DIR, f"{safe_title}_global_seismograph.csv")
    df.to_csv(csv_out, index=False)
    print(f"\n[Output] Saved global analysis to {csv_out}")
    
    # Also save correlations into a JSON sidecar
    corr_out = os.path.join(OUTPUT_DIR, f"{safe_title}_correlations.json")
    with open(corr_out, "w", encoding="utf-8") as f:
        json.dump({
            "pearson_r": res["pearson_r"],
            "spearman_rho": res["spearman_rho"],
            "n_regimes": res["n_regimes"]
        }, f, indent=2)

    # Generate HTML Dashboard
    html_out = os.path.join(OUTPUT_DIR, f"{safe_title}_seismograph_strict.html")
    report_title = " + ".join(args.book)
    if args.limit:
        report_title += f" (first {args.limit} verses)"
    html_str = build_report_html(df, report_title, z_threshold=2.0, pearson_r=res["pearson_r"], spearman_rho=res["spearman_rho"])
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"[Output] Generated Interactive Dashboard at {html_out}")

if __name__ == "__main__":
    main()
