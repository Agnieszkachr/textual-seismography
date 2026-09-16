#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction probe: does a masked model put verses back word for word?

A model that has only learned the language guesses plausible Hebrew. A model
that has read the text returns the original words. The probe hides part of a
verse and measures how much comes back verbatim.

Three controls keep the comparison honest:

  * the masks are drawn once per verse and condition and reused for every
    model, so the two models answer identical questions;
  * word-level scoring is also reported over the subset of words that both
    tokenisers encode as a single token, because the tokenisers do not split
    Hebrew equally and whole-word accuracy would otherwise favour the coarser
    one;
  * verses are selected by each model's own perplexity as well as by the
    other's, so neither is only tested on ground the other chose.

The number of mask tokens reveals how many sub-word pieces the hidden word
has. That makes the task easier than open-ended generation, and the absolute
rates should be read with that in mind; it applies equally to both models.

    python extraction_probe.py --metric output/isaiah_abstract_metric.csv \
        --berel-pll output/berel_pll.json --out output/extraction_probe.csv
"""
import argparse, json, random, time
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODELS = {"BEREL 3.0": "dicta-il/BEREL_3.0", "DictaBERT": "dicta-il/dictabert"}
SEED = 20260916
CONDITIONS = [("1 word", "scatter", 0), ("25%", "scatter", .25),
              ("50%", "scatter", .50), ("50% contiguous", "block", .50)]


def choose(words, kind, frac, rng):
    n = len(words)
    k = 1 if frac == 0 else max(1, round(n * frac))
    if kind == "block":
        return set(range(n - k, n))
    return set(rng.sample(range(n), k))


@torch.no_grad()
def recover(model, tok, words, hide):
    gold, spans, pieces = [], [], []
    for i, w in enumerate(words):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if i in hide:
            spans.append((i, len(gold), len(gold) + len(ids)))
            gold.extend(ids)
            pieces.append([tok.mask_token_id] * len(ids))
        else:
            pieces.append(ids)
    flat, mask_pos = [tok.cls_token_id], []
    for i, p in enumerate(pieces):
        if i in hide:
            mask_pos.extend(range(len(flat), len(flat) + len(p)))
        flat.extend(p)
    flat.append(tok.sep_token_id)
    ids = torch.tensor([flat])
    todo = set(mask_pos)
    while todo:                                  # commit the surest mask, re-predict
        probs = torch.softmax(model(ids).logits[0], -1)
        best_p, best_i, best_t = -1.0, None, None
        for i in todo:
            p, t = probs[i].max(-1)
            if p.item() > best_p:
                best_p, best_i, best_t = p.item(), i, t.item()
        ids[0, best_i] = best_t
        todo.discard(best_i)
    out = ids[0, mask_pos].tolist()
    return {wi: out[a:b] == gold[a:b] for wi, a, b in spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="output/isaiah_abstract_metric.csv")
    ap.add_argument("--berel-pll", default="output/berel_pll.json")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="output/extraction_probe.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.metric)
    d["ppl_berel"] = d.verse_id.map(json.load(open(a.berel_pll)))
    d["nw"] = d.text.astype(str).str.split().str.len()
    d = d[d.nw.between(8, 16)].copy()

    lo_b, hi_b = d.ppl_berel.quantile([.4, .6])
    lo_d, hi_d = d.ppl_dicta.quantile([.4, .6])
    sets = {
        "lowest perplexity, BEREL":     d.nsmallest(a.n, "ppl_berel"),
        "lowest perplexity, DictaBERT": d.nsmallest(a.n, "ppl_dicta"),
        "control, both at median":      d[d.ppl_berel.between(lo_b, hi_b) &
                                          d.ppl_dicta.between(lo_d, hi_d)]
                                         .sample(a.n, random_state=SEED),
    }

    toks = {m: AutoTokenizer.from_pretrained(p) for m, p in MODELS.items()}

    def single(w):
        return all(len(t(w, add_special_tokens=False)["input_ids"]) == 1 for t in toks.values())

    for m, t in toks.items():
        allw = [w for txt in d.text.astype(str) for w in txt.split()]
        n = [len(t(w, add_special_tokens=False)["input_ids"]) for w in allw]
        print(f"  {m:10s} {sum(n)/len(n):.2f} pieces per word", flush=True)

    plan = {}
    for sname, frame in sets.items():
        for r in frame.itertuples():
            w = str(r.text).split()
            rng = random.Random(f"{SEED}-{r.verse_id}")
            for cname, kind, frac in CONDITIONS:
                plan[(sname, r.verse_id, cname)] = (w, choose(w, kind, frac, rng))

    rows = []
    for mname, mpath in MODELS.items():
        model = AutoModelForMaskedLM.from_pretrained(mpath).eval()
        tok = toks[mname]
        t0 = time.time()
        for (sname, vid, cname), (words, hide) in plan.items():
            got = recover(model, tok, words, hide)
            for wi, ok in got.items():
                rows.append({"model": mname, "set": sname, "condition": cname,
                             "verse": vid, "word": words[wi],
                             "ok": int(ok), "single_token": int(single(words[wi]))})
        print(f"  {mname:10s} done in {time.time()-t0:.0f}s", flush=True)
        del model

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    for label, sel in (("all words", out), ("words both models keep whole", out[out.single_token == 1])):
        p = (sel.groupby(["model", "set", "condition"]).ok.mean() * 100).unstack()
        print(f"\n=== verbatim recovery, {label} (%) ===")
        print(p[[c[0] for c in CONDITIONS]].round(1).to_string())


if __name__ == "__main__":
    main()
