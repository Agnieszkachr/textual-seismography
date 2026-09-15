"""
DataManager — Handles fetching, caching, parsing, and cleaning of
Westminster Leningrad Codex (WLC) OSIS XML files from OpenScriptures,
and saving analysis results to CSV.
"""

import os
import re
import requests
import pandas as pd
from lxml import etree

# ---------------------------------------------------------------------------
# Book URL registry (expandable)
# ---------------------------------------------------------------------------
BOOK_URLS: dict[str, str] = {
    "Genesis":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Gen.xml",
    "Exodus":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Exod.xml",
    "Leviticus":   "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Lev.xml",
    "Numbers":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Num.xml",
    "Deuteronomy": "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Deut.xml",
    "Joshua":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Josh.xml",
    "Judges":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Judg.xml",
    "1 Samuel":    "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/1Sam.xml",
    "2 Samuel":    "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/2Sam.xml",
    "1 Kings":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/1Kgs.xml",
    "2 Kings":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/2Kgs.xml",
    "Isaiah":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Isa.xml",
    "Jeremiah":    "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Jer.xml",
    "Ezekiel":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Ezek.xml",
    "Psalms":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Ps.xml",
    "Job":         "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Job.xml",
    "Proverbs":    "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Prov.xml",
    "Ruth":        "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Ruth.xml",
    "Song of Solomon": "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Song.xml",
    "Ecclesiastes":"https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Eccl.xml",
    "Lamentations":"https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Lam.xml",
    "Esther":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Esth.xml",
    "Daniel":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Dan.xml",
    "Ezra":        "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Ezra.xml",
    "Nehemiah":    "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Neh.xml",
    "1 Chronicles":"https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/1Chr.xml",
    "2 Chronicles":"https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/2Chr.xml",
    "Hosea":       "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Hos.xml",
    "Joel":        "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Joel.xml",
    "Amos":        "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Amos.xml",
    "Obadiah":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Obad.xml",
    "Jonah":       "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Jonah.xml",
    "Micah":       "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Mic.xml",
    "Nahum":       "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Nah.xml",
    "Habakkuk":    "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Hab.xml",
    "Zephaniah":   "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Zeph.xml",
    "Haggai":      "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Hag.xml",
    "Zechariah":   "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Zech.xml",
    "Malachi":     "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc/Mal.xml",
}

DEFAULT_BOOK = "Genesis"

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# OSIS XML namespace
OSIS_NS = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}


class DataManager:
    """Fetch, cache, parse, clean WLC texts and save analysis results."""

    def __init__(self) -> None:
        os.makedirs(SRC_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetch / Cache
    # ------------------------------------------------------------------
    def get_text(self, book_name: str = DEFAULT_BOOK) -> str:
        """Return raw XML string for *book_name*, downloading only if not cached."""
        if book_name not in BOOK_URLS:
            raise ValueError(
                f"Unknown book '{book_name}'. Available: {list(BOOK_URLS.keys())}"
            )

        cache_path = os.path.join(SRC_DIR, f"{book_name}.xml")

        if os.path.isfile(cache_path):
            print(f"[DataManager] Loading cached '{book_name}' from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as fh:
                return fh.read()

        url = BOOK_URLS[book_name]
        print(f"[DataManager] Downloading '{book_name}' from {url} …")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
        print(f"[DataManager] Saved to {cache_path}")
        return resp.text

    # ------------------------------------------------------------------
    # OSIS XML → verse list
    # ------------------------------------------------------------------
    def parse_verses(self, xml_str: str) -> list[dict]:
        """Parse OSIS XML and return [{"verse_id": "Gen.1.1", "text": "…"}, …]."""
        root = etree.fromstring(xml_str.encode("utf-8"))
        verses: list[dict] = []

        # Find all <verse> elements (may be nested inside <chapter>)
        for verse_el in root.iter("{http://www.bibletechnologies.net/2003/OSIS/namespace}verse"):
            osis_id = verse_el.get("osisID")
            if osis_id is None:
                continue  # closing milestone, skip

            # Collect word texts from <w> children (recursive)
            words: list[str] = []
            for w_el in verse_el.iter("{http://www.bibletechnologies.net/2003/OSIS/namespace}w"):
                if w_el.text:
                    words.append(w_el.text.strip())

            if words:
                verses.append({
                    "verse_id": osis_id,
                    "text": " ".join(words),
                })

        print(f"[DataManager] Parsed {len(verses)} verses.")
        return verses

    # ------------------------------------------------------------------
    # Hebrew text cleaner (consonants only)
    # ------------------------------------------------------------------
    @staticmethod
    def clean_hebrew(text: str) -> str:
        """Strip ALL niqqud & cantillation marks.  Keep ONLY Hebrew consonants
        (abjad U+05D0-U+05EA), spaces, and maqaf (U+05BE, the Hebrew hyphen).

        The Hebrew GPT model expects modern unvocalized Hebrew.
        """
        # 1. Remove cantillation / te'amim (U+0591 – U+05AF)
        text = re.sub(r"[\u0591-\u05AF]", "", text)
        # 2. Remove niqqud / vowel points (U+05B0 – U+05BD, U+05BF, U+05C1, U+05C2, U+05C4, U+05C5, U+05C7)
        text = re.sub(r"[\u05B0-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]", "", text)
        # 3. Keep only Hebrew consonants, maqaf, and whitespace
        text = re.sub(r"[^\u05D0-\u05EA\u05BE\s]", "", text)
        # 4. Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def save_results(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to ./output/{filename}.csv (UTF-8 BOM for Excel).
        Returns the absolute path of the saved file.
        """
        if not filename.endswith(".csv"):
            filename += ".csv"
        out_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[DataManager] Results saved to {out_path}")
        return out_path
