"""Pre-push privacy check: is anything private about to be published?

Run locally before pushing (the repository is public):   PYTHONPATH=src .venv/bin/python scripts/check_public.py

Scans every file git would publish (tracked files plus new, non-ignored files) for:
  - the ESPN login cookies and league id from .env
  - ESPN member ids ({XXXXXXXX-XXXX-...} GUIDs)
  - other managers' names and the league's fantasy team names (from the local data cache, if present)
Exits with status 1 and lists the hits if anything is found.
"""

from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GUID = re.compile(r"\{?[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\}?", re.I)
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".pkl", ".parquet"}


def publishable_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files = [ROOT / f for f in out.splitlines()]
    return [f for f in files if f.is_file() and not f.is_symlink() and f.suffix.lower() not in SKIP_SUFFIX]


def private_terms() -> dict[str, str]:
    from iknowpuck.config import CACHE_DIR, load_credentials

    creds = load_credentials()
    terms: dict[str, str] = {}
    if creds.league_id:
        terms[str(creds.league_id)] = "league id"
    for label, value in (("espn_s2 cookie", creds.espn_s2), ("SWID cookie", creds.swid)):
        if value:
            terms[value[:16]] = label
            terms[value.strip("{}")[:12]] = label
    bundles = sorted(Path(CACHE_DIR).glob("bundle_*.pkl"))
    if bundles:
        with open(bundles[-1], "rb") as fh:  # written only by this project on this machine
            b = pickle.load(fh)
        me = str(b.manager_of_team.get(b.settings.my_team_id))
        author = set(b.managers.loc[b.managers.owner_id.astype(str) == me, "manager"]) if len(b.managers) else set()
        for name in set(b.managers.manager) - author:
            terms[name] = "manager name"
            last = name.split()[-1]
            if len(last) >= 5:
                terms[last] = "manager surname"
        for t in b.settings.team_names.values():
            if t and len(t) >= 6:
                terms[t] = "fantasy team name"
    return terms


def main() -> int:
    terms = private_terms()
    hits: list[str] = []
    for f in publishable_files():
        if f.name == "check_public.py":
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = f.relative_to(ROOT)
        for term, label in terms.items():
            if term and term in text:
                hits.append(f"{rel}: {label}")
        if GUID.search(text):
            hits.append(f"{rel}: ESPN member id (GUID)")
    if hits:
        print("Private data found in files that would be published:")
        for h in sorted(set(hits)):
            print("  " + h)
        return 1
    print(f"OK: {len(publishable_files())} publishable files checked against {len(terms)} private terms; nothing private found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
