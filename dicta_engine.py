"""
DictaEngine — Detects narrative fractures using DictaBERT (Masked LM)
by computing per-verse Pseudo-Log-Likelihood (PLL) and normalising
the signal as Z-scores.

Mirrors the SeismographEngine interface so both engines can be used
interchangeably in the ensemble pipeline.
"""

import sys
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

DICTA_MODEL = "dicta-il/dictabert"
DICTA_REVISION = "8884c6db002aba4002ee638fe4070c92e9ffbbf1"


class DictaEngine:
    """Load DictaBERT (masked LM) and score verse transitions via PLL."""

    def __init__(self, model_name: str = DICTA_MODEL) -> None:
        print(f"[DictaEngine] Loading {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=DICTA_REVISION)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, revision=DICTA_REVISION)
        print(f"[DictaEngine] DictaBERT loaded revision: {getattr(self.model.config, '_commit_hash', 'unknown')}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"[DictaEngine] Model loaded on {self.device}.")

    # ------------------------------------------------------------------
    # Core: Pseudo-Log-Likelihood for one text
    # ------------------------------------------------------------------
    def _compute_pll(self, text: str) -> float:
        """Compute Pseudo-Log-Likelihood (PLL) for a text string.

        For each token, mask it and measure how well the model predicts it
        given all other tokens. Returns exp(-mean_log_prob) as a
        perplexity-like score (higher = more surprising).
        """
        if not text.strip():
            return float("nan")

        ids = self.tokenizer.encode(text, add_special_tokens=True,
                                    truncation=True, max_length=512)
        input_tensor = torch.tensor([ids], device=self.device)

        # Identify maskable positions (skip [CLS], [SEP], [PAD])
        special = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                   self.tokenizer.pad_token_id}
        positions = [i for i, tok_id in enumerate(ids) if tok_id not in special]

        if not positions:
            return float("nan")

        num_masks = len(positions)
        
        # Create a batch of identical sequences
        # Shape: (num_masks, seq_length)
        masked_batch = input_tensor.repeat(num_masks, 1)
        
        # Batch collect the original target IDs
        target_ids = []
        for i, pos in enumerate(positions):
            target_ids.append(masked_batch[i, pos].item())
            masked_batch[i, pos] = self.tokenizer.mask_token_id

        log_probs = []

        with torch.no_grad():
            # Process the entire batch of masked sequences in one forward pass
            outputs = self.model(input_ids=masked_batch)
            
            # For each sequence in the batch, extract the logits at the masked position
            for i, pos in enumerate(positions):
                logits = outputs.logits[i, pos]  # vocab-size vector
                probs = torch.softmax(logits, dim=-1)
                original_id = target_ids[i]
                lp = torch.log(probs[original_id] + 1e-12).item()
                log_probs.append(lp)

        mean_log_prob = sum(log_probs) / len(log_probs)
        return math.exp(-mean_log_prob)

    # ------------------------------------------------------------------
    # Full pipeline (matches SeismographEngine.calculate_fractures)
    # ------------------------------------------------------------------
    def calculate_fractures(
        self,
        verses: list[dict],
        context_window: int = 3,
        rolling_window: int = 5,
        progress_callback=None,
    ) -> pd.DataFrame:
        """Score every verse and return a DataFrame with Z-scores.

        Parameters
        ----------
        verses : list[dict]
            Each dict has ``verse_id`` (str) and ``text`` (str, cleaned).
        context_window : int
            Unused — kept for interface compatibility with SeismographEngine.
        rolling_window : int
            Window size for rolling mean / std / Z-score.
        progress_callback : callable, optional
            Called with (current_index, total) for progress reporting.

        Returns
        -------
        pd.DataFrame
            Columns: verse_id, text_snippet, perplexity_score,
            rolling_mean, rolling_std, z_score.
        """
        total = len(verses)
        records: list[dict] = []

        print(f"[DictaEngine] Scoring {total} verses …")
        for idx, v in enumerate(verses):
            ppl = self._compute_pll(v["text"])
            records.append({
                "verse_id": v["verse_id"],
                "text_snippet": v["text"][:80],
                "perplexity_score": ppl,
            })
            if progress_callback:
                progress_callback(idx + 1, total)
            elif (idx + 1) % 50 == 0 or idx == total - 1:
                print(f"\r  … {idx + 1}/{total}", end="", flush=True)

        if not progress_callback:
            print()  # newline after progress

        df = pd.DataFrame(records)

        # Rolling statistics & Z-score (identical to SeismographEngine)
        df["rolling_mean"] = (
            df["perplexity_score"]
            .rolling(window=rolling_window, min_periods=1, center=True)
            .mean()
        )
        df["rolling_std"] = (
            df["perplexity_score"]
            .rolling(window=rolling_window, min_periods=1, center=True)
            .std(ddof=0)
        )
        df["z_score"] = df.apply(
            lambda row: (
                (row["perplexity_score"] - row["rolling_mean"]) / row["rolling_std"]
                if row["rolling_std"] > 0
                else 0.0
            ),
            axis=1,
        )

        return df
