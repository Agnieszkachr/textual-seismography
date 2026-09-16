#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Permutation validation for the two-witness rule.

The claim under test is not "these verses are odd" but "the two models agree on
which verses are odd more often than they would by accident". This script
measures that, under three nulls of increasing strictness, and reports the
verse-length confound explicitly.

    python validate_agreement.py --metric output/isaiah_abstract_metric.csv
    python validate_agreement.py --metric output/isaiah_abstract_metric.csv \
                                 --extra output/isaiah_berel_check.csv:z_berel

Nulls
  rotate   circular shift of one model's score series: keeps the
           autocorrelation the rolling window induces, breaks the pairing
  shuffle  free permutation: breaks autocorrelation too, so it is the weaker
           of the two and is reported only for comparison
  length   permutation *within* verse-length deciles: the length-to-score
           relationship is preserved in the null, so any excess agreement
           that survives cannot be attributed to verse length

Outputs a table and, with --json, a machine-readable summary.
"""
import argparse, json, sys
import numpy as np
import pandas as pd

Z_THRESHOLD = 1.9
N_DRAWS = 20000
SEED = 20260916


def shared(a, b, zt):
    return int(((a >= zt) & (b >= zt)).sum())


def null_distribution(zg, zm, kind, bins, n_draws, rng, zt):
    hits = np.empty(n_draws, dtype=int)
    n = len(zm)
    if kind == "length":
        groups = [np.where(bins == k)[0] for k in np.unique(bins)]
    for i in range(n_draws):
        if kind == "rotate":
            zz = np.roll(zm, rng.integers(1, n))
        elif kind == "shuffle":
            zz = rng.permutation(zm)
        elif kind == "length":
            zz = zm.copy()
            for ix in groups:
                zz[ix] = zm[rng.permutation(ix)]
        else:
            raise ValueError(kind)
        hits[i] = shared(zg, zz, zt)
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", required=True,
                    help="CSV with verse_id, text, z_gpt and at least one masked-model z column")
    ap.add_argument("--extra", action="append", default=[],
                    help="additional masked model as PATH:COLUMN, joined on verse_id")
    ap.add_argument("--causal", default="z_gpt")
    ap.add_argument("--masked", action="append", default=None,
                    help="masked z columns to test (default: every z_* column except the causal one)")
    ap.add_argument("--threshold", type=float, default=Z_THRESHOLD)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", metavar="PATH", help="write the summary here as well")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    d = pd.read_csv(a.metric)
    for spec in a.extra:
        path, _, col = spec.rpartition(":")
        add = pd.read_csv(path).set_index("verse_id")[col]
        d[col] = add.reindex(d.verse_id).values

    masked = a.masked or [c for c in d.columns
                          if c.startswith("z_") and c != a.causal and d[c].notna().any()]
    if "text" not in d.columns:
        sys.exit("the metric file needs a 'text' column for the length-matched null")

    zg = d[a.causal].to_numpy(float)
    nword = d.text.astype(str).str.split().str.len().to_numpy()
    bins = np.asarray(pd.qcut(nword, 10, labels=False, duplicates="drop"))
    zt = a.threshold

    print(f"verses {len(d)} | threshold z >= {zt} | {a.draws} draws | seed {a.seed}")
    print(f"{a.causal} peaks: {int((zg >= zt).sum())}\n")
    print(f"{'masked model':<12}{'peaks':>7}{'shared':>8}   {'null':<8}"
          f"{'mean':>7}{'95th':>6}{'max':>5}{'p':>10}")

    out = {"threshold": zt, "draws": a.draws, "seed": a.seed, "models": {}}
    for m in masked:
        zm = d[m].to_numpy(float)
        obs = shared(zg, zm, zt)
        rec = {"peaks": int((zm >= zt).sum()), "shared": obs, "nulls": {}}
        for kind in ("rotate", "shuffle", "length"):
            h = null_distribution(zg, zm, kind, bins, a.draws, rng, zt)
            p = float((h >= obs).mean())
            rec["nulls"][kind] = {"mean": float(h.mean()),
                                  "pct95": float(np.percentile(h, 95)),
                                  "max": int(h.max()),
                                  "p": p,
                                  "p_is_upper_bound": p == 0.0}
            shown = f"<{1/a.draws:.0e}" if p == 0 else f"{p:.5f}"
            head = f"{m:<12}{rec['peaks']:>7}{obs:>8}   " if kind == "rotate" else " " * 27 + "   "
            print(f"{head}{kind:<8}{h.mean():>7.2f}{np.percentile(h,95):>6.0f}"
                  f"{h.max():>5}{shown:>10}")
        out["models"][m] = rec
        print()

    # the confound, stated rather than hidden
    from scipy import stats
    print("verse length (words) vs score, Spearman rho:")
    for c in [a.causal] + list(masked):
        r, p = stats.spearmanr(d[c], nword, nan_policy="omit")
        print(f"  {c:<12} rho = {r:+.3f}  p = {p:.3g}")
        out.setdefault("length_confound", {})[c] = {"rho": float(r), "p": float(p)}
    first = masked[0]
    seam = (zg >= zt) & (d[first].to_numpy(float) >= zt)
    u = stats.mannwhitneyu(nword[seam], nword[~seam])
    print(f"  shared seams average {nword[seam].mean():.1f} words vs "
          f"{nword[~seam].mean():.1f} elsewhere (Mann-Whitney p = {u.pvalue:.4f})")
    out["length_confound"]["seam_words"] = float(nword[seam].mean())
    out["length_confound"]["other_words"] = float(nword[~seam].mean())
    out["length_confound"]["mannwhitney_p"] = float(u.pvalue)

    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"\nwritten: {a.json}")


if __name__ == "__main__":
    main()
