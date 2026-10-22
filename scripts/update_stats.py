#!/usr/bin/env python3
"""Refresh stats.json with all-time download counts.

Reads the "hf" (Hugging Face repo id) and "pypi" (package name) fields from
links.json, plus likes for the "hf" repos, plus the GitHub repo behind every entry with a "github" icon (taken
from its url) for star counts. If a lookup fails, the previous value is kept so a flaky API
never blanks a number on the page.
"""
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_json(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "nxai-linktree-stats", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res)


def hf_downloads(repo):
    return get_json(f"https://huggingface.co/api/models/{repo}?expand[]=downloadsAllTime")["downloadsAllTime"]


def hf_likes(repo):
    return get_json(f"https://huggingface.co/api/models/{repo}?expand[]=likes")["likes"]


def pypi_downloads(package):
    return get_json(f"https://pepy.tech/api/v2/projects/{urllib.parse.quote(package)}")["total_downloads"]


def github_stars(repo):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return get_json(f"https://api.github.com/repos/{repo}", headers)["stargazers_count"]


def github_repo(link):
    m = re.match(r"https://github\.com/([^/]+/[^/#?]+)", link.get("url", ""))
    return m.group(1) if link.get("icon") == "github" and m else None


def main():
    links = json.loads((ROOT / "links.json").read_text())
    stats_path = ROOT / "stats.json"
    old = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for l in links:
        repo = github_repo(l)
        if repo:
            l["github"] = repo
    stats = {"hf": {}, "pypi": {}, "hf_likes": {}, "github": {}}

    for source, field, fetch in (("hf", "hf", hf_downloads), ("hf_likes", "hf", hf_likes), ("pypi", "pypi", pypi_downloads), ("github", "github", github_stars)):
        for key in sorted({l[field] for l in links if field in l}):
            try:
                stats[source][key] = fetch(key)
            except Exception as err:
                print(f"warning: {source} {key}: {err}")
                if key in old.get(source, {}):
                    stats[source][key] = old[source][key]

    stats["updated"] = date.today().isoformat()
    if {k: v for k, v in stats.items() if k != "updated"} == {k: v for k, v in old.items() if k != "updated"}:
        return
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
