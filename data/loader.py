"""
Download and parse the McAuley UCSD Steam dataset.

Files:
  australian_user_reviews.json.gz , user_id, reviews[{item_id, recommend, review, posted}]
  australian_users_items.json.gz  , user_id, items[{item_id, playtime_forever, playtime_2weeks}]

Both are newline-delimited JSON (one Python dict literal per line).
"""
import gzip
import ast
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REVIEWS_URL, ITEMS_URL, REVIEWS_PATH, ITEMS_PATH


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def _parse_gz(path: Path) -> list[dict]:
    """Parse newline-delimited Python dict literals from a .json.gz file."""
    records = []
    skipped = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ast.literal_eval(line))
            except Exception:
                skipped += 1
                continue
    if skipped:
        # Don't drop malformed lines silently; an upstream format change would
        # otherwise make rows disappear with no trace.
        print(f"  Skipped {skipped} unparseable line(s) in {path.name}")
    return records


def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    """
    Returns flat DataFrame:
      user_id, item_id, recommend (bool), review_text (str), posted (str)
    """
    raw = _parse_gz(path)
    rows = []
    for user_rec in raw:
        uid = user_rec.get("user_id", "")
        for rev in user_rec.get("reviews", []):
            rows.append({
                "user_id":     uid,
                "item_id":     str(rev.get("item_id", "")),
                "recommend":   bool(rev.get("recommend", False)),
                "review_text": str(rev.get("review", "")).strip(),
                "posted":      str(rev.get("posted", "")),
            })
    df = pd.DataFrame(rows)
    # Drop empty reviews, text is required for LLM extraction
    df = df[df["review_text"].str.len() > 20].reset_index(drop=True)
    return df


def load_items(path: Path = ITEMS_PATH) -> pd.DataFrame:
    """
    Returns flat DataFrame:
      user_id, item_id, playtime_forever (float, hours), playtime_2weeks (float)
    """
    raw = _parse_gz(path)
    rows = []
    for user_rec in raw:
        uid = user_rec.get("user_id", "")
        items_count = int(user_rec.get("items_count", 0))
        for item in user_rec.get("items", []):
            rows.append({
                "user_id":        uid,
                "item_id":        str(item.get("item_id", "")),
                "playtime_forever": float(item.get("playtime_forever", 0)) / 60,  # convert min → hours
                "playtime_2weeks":  float(item.get("playtime_2weeks", 0)) / 60,
                "items_count":    items_count,
            })
    return pd.DataFrame(rows)


def download_all() -> None:
    _download(REVIEWS_URL, REVIEWS_PATH)
    _download(ITEMS_URL, ITEMS_PATH)
