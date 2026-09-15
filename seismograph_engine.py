"""
SeismographEngine — Detects narrative fractures (editorial splices) in
Hebrew texts by computing per-verse-transition perplexity with a
Hebrew Causal LM and normalising the signal as Z-scores.
"""

import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Norod78/hebrew-gpt_neo-small"
GPT_REVISION = "d9ee75ea9eb03f817c5c97a4e4d7b2e300be365b"


class SeismographEngine:
    """Load a Hebrew causal LM and score verse transitions."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        print(f"[SeismographEngine] Loading model '{model_name}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=GPT_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=GPT_REVISION)
        print(f"[SeismographEngine] GPT-Neo loaded revision: {getattr(self.model.config, '_commit_hash', 'unknown')}")

        # Ensure pad_token exists (GPT-Neo may not set one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"[SeismographEngine] Model loaded on {self.device}.")

    # ------------------------------------------------------------------
    # Core: score one transition with a sliding context window
    # ------------------------------------------------------------------
    def _score_transition(
        self,
        units: list[str],
        target_idx: int,
        context_window: int = 3,
    ) -> float:
        """Compute the perplexity of *units[target_idx]* conditioned on
        up to *context_window* preceding units.

        Returns exp(cross-entropy loss on target tokens only).
        """
        # Build context + target strings
        start = max(0, target_idx - context_window)
        context_text = " ".join(units[start:target_idx])
        target_text = units[target_idx]

        if not target_text.strip():
            return float("nan")

        # Tokenise separately to know the boundary
        ctx_ids = self.tokenizer.encode(context_text, add_special_tokens=False) if context_text else []
        tgt_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

        if not tgt_ids:
            return float("nan")

        input_ids = torch.tensor([ctx_ids + tgt_ids], device=self.device)

        # Build labels: -100 for context tokens (ignored), real ids for target
        labels = input_ids.clone()
        if ctx_ids:
            labels[0, : len(ctx_ids)] = -100

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss  # mean cross-entropy over non-ignored tokens

        if loss is None or math.isnan(loss.item()):
            return float("nan")

        return math.exp(loss.item())

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def calculate_fractures(
        self,
        verses: list[dict],
        context_window: int = 3,
        rolling_window: int = 5,
        progress_callback=None,
    ) -> pd.DataFrame:
        """Score every verse transition and return a DataFrame with Z-scores.

        Parameters
        ----------
        verses : list[dict]
            Each dict has ``verse_id`` (str) and ``text`` (str, cleaned).
        context_window : int
            Number of preceding verses used as context.
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
        texts = [v["text"] for v in verses]
        records: list[dict] = []

        total = len(verses)
        for idx in range(total):
            ppl = self._score_transition(texts, idx, context_window)
            snippet = verses[idx]["text"][:80]
            records.append({
                "verse_id": verses[idx]["verse_id"],
                "text_snippet": snippet,
                "perplexity_score": ppl,
            })
            if progress_callback:
                progress_callback(idx + 1, total)

        df = pd.DataFrame(records)

        # Rolling statistics & Z-score
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
        # Avoid division by zero
        df["z_score"] = df.apply(
            lambda row: (
                (row["perplexity_score"] - row["rolling_mean"]) / row["rolling_std"]
                if row["rolling_std"] > 0
                else 0.0
            ),
            axis=1,
        )

        return df
