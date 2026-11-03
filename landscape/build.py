#!/usr/bin/env python3
"""Build the inference landscape image from landscape.toml.

Checks every tile against README.md, reports README sections with no tile,
renders an HTML grid and screenshots it with headless Chromium.

    python3 landscape/build.py                  # check + render to landscape/build/
    python3 landscape/build.py --fetch-logos    # download missing GitHub avatars first
    python3 landscape/build.py --check-only     # validate and report coverage only
    python3 landscape/build.py --out assets/inference-landscape.png

Standard library only; needs a Chromium or Chrome binary for rendering.
"""
import argparse
import html
import re
import shutil
import socket
import subprocess
import sys
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGOS = HERE / "logos"
BUILD = HERE / "build"
PAGE_WIDTH = 1900  # CSS px; rendered at 2x for a 3800 px wide PNG
SCALE = 2


# ---------------------------------------------------------------- README model

def parse_readme(path):
    """Return (entry URL -> subsection anchor, set of anchors, {number: H2 title}, [(anchor, H3 title)])."""
    entries, anchors, h2, h3 = {}, set(), {}, []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r'<a id="([^"]+)"></a>', line)
        if m:
            anchors.add(m[1])
            current = m[1]
            continue
        m = re.match(r"## (\d+)\. (.+)", line)
        if m:
            h2[int(m[1])] = m[2].strip()
            continue
        m = re.match(r"### (.+)", line)
        if m:
            h3.append((current, m[1].strip()))
            continue
        m = re.match(r"- \[[^\]]+\]\((https?://[^)\s]+)\)", line)
        if m:
            entries[m[1]] = current
    return entries, anchors, h2, h3


# ---------------------------------------------------------------- data model

def gh_owner(link):
    m = re.match(r"https://github\.com/([^/]+)/", link)
    return m[1] if m else None


def logo_file(tile):
    """Resolve a tile's logo to (path, source URL or None), or (None, None)."""
    logo = tile.get("logo")
    if not logo:
        return None, None
    if logo == "gh" or logo.startswith("gh:"):
        owner = logo[3:] if logo.startswith("gh:") else gh_owner(tile["link"])
        if not owner:
            return None, None
        return LOGOS / f"gh-{owner}.png", f"https://github.com/{owner}.png?size=200"
    return LOGOS / logo, None


def default_badge(name):
    words = re.findall(r"[A-Za-z0-9]+", name)
    return "".join(w[0] for w in words[:3]).upper() if len(words) > 1 else name[:4]


def check(data, readme):
    """Validate tiles against the README. Returns (errors, warnings, tile anchors per README subsection)."""
    entries, anchors, h2, _ = readme
    errors, warnings, covered = [], [], {}
    seen = {}
    for sec in data["section"]:
        want = h2.get(sec["number"])
        if want != sec["title"]:
            errors.append(f"section {sec['number']} is titled {sec['title']!r}, README has {want!r}")
        for card in sec.get("card", []):
            for tile in card["tiles"]:
                name, link = tile["name"], tile["link"]
                where = f"{sec['number']}/{card['title']}/{name}"
                if link.startswith("#"):
                    if link[1:] not in anchors:
                        errors.append(f"{where}: anchor {link} not in README")
                    else:
                        covered.setdefault(link[1:], []).append(name)
                elif link not in entries:
                    errors.append(f"{where}: link is not a README entry: {link}")
                else:
                    covered.setdefault(entries[link], []).append(name)
                if link in seen:
                    warnings.append(f"{where}: same link as {seen[link]}")
                seen[link] = where
                path, _ = logo_file(tile)
                if tile.get("logo") and (path is None or not path.exists()):
                    warnings.append(f"{where}: logo {tile['logo']!r} missing, using badge")
    return errors, warnings, covered


def _download(path, url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "landscape-build"})
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
        print(f"fetched {path.name}")
    except Exception as e:  # keep going; the tile falls back to its badge
        print(f"could not fetch {url}: {e}", file=sys.stderr)


def fetch_logos(data):
    LOGOS.mkdir(exist_ok=True)
    # urllib has no happy-eyeballs, so a broken IPv6 route costs a full timeout per request.
    getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = lambda *a, **k: sorted(getaddrinfo(*a, **k), key=lambda ai: ai[0] != socket.AF_INET)
    sources, todo = {}, {}
    for sec in data["section"]:
        for card in sec.get("card", []):
            for tile in card["tiles"]:
                path, url = logo_file(tile)
                if not url:
                    continue
                sources[path.name] = url
                if not path.exists():
                    todo[path] = url
    with ThreadPoolExecutor(max_workers=16) as pool:
        for path, url in todo.items():
            pool.submit(_download, path, url)
    socket.getaddrinfo = getaddrinfo
    # Keep hand-added entries for logo files that are not GitHub avatars.
    listing = LOGOS / "SOURCES.txt"
    if listing.exists():
        for line in listing.read_text(encoding="ascii").splitlines():
            name = line.split("\t", 1)[0]
            if "\t" in line and not line.startswith("#") and not name.startswith("gh-"):
                sources.setdefault(name, line.split("\t", 1)[1])
    lines = [f"{name}\t{url}" for name, url in sorted(sources.items()) if (LOGOS / name).exists()]
    (LOGOS / "SOURCES.txt").write_text(
        "# logo file<TAB>source. Logos are trademarks of their respective owners.\n" + "\n".join(lines) + "\n",
        encoding="ascii",
    )


# ---------------------------------------------------------------- rendering

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { width: %(w)dpx; background: #0a0e17; color: #e5e7eb; padding: 36px 32px 28px;
       font-family: Inter, "Noto Sans", "Liberation Sans", Arial, sans-serif; }
h1 { text-align: center; font-size: 60px; font-weight: 800; color: #fff; margin-bottom: 28px; letter-spacing: -0.5px; }
.section { margin-bottom: 26px; }
.shead { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
.num { width: 40px; height: 40px; border-radius: 50%%; display: flex; align-items: center; justify-content: center;
       font-weight: 700; font-size: 20px; color: #0a0e17; }
.shead h2 { font-size: 36px; font-weight: 800; color: #fff; white-space: nowrap; }
.shead .rule { flex: 1; height: 2px; opacity: .55; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.card { background: #111725; border-radius: 10px; border-top: 2px solid; padding: 14px 10px 16px; }
.card h3 { font-size: 15px; font-weight: 800; text-transform: uppercase; letter-spacing: .3px; margin: 0 4px 12px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); row-gap: 14px; }
.tile { display: flex; flex-direction: column; align-items: center; text-align: center; }
.icon { width: 78px; height: 78px; border-radius: 14px; background: #f8fafc; display: flex; flex-direction: column;
        align-items: center; justify-content: center; overflow: hidden; }
.icon img { width: 78px; height: 78px; object-fit: contain; }
.badge { font-weight: 800; line-height: 1; }
.sub { font-size: 10px; font-weight: 600; color: #6b7280; margin-top: 4px; }
.label { font-size: 15px; font-weight: 700; color: #f3f4f6; margin-top: 8px; line-height: 1.2; max-width: 104px; }
footer { display: flex; justify-content: space-between; font-size: 15px; color: #9ca3af; margin-top: 6px; }
"""


def badge_size(text):
    return {1: 34, 2: 30, 3: 27, 4: 23, 5: 20}.get(len(text), 16)


def render_tile(tile, color):
    path, _ = logo_file(tile)
    if path is not None and path.exists():
        icon = f'<img src="{path.as_uri()}" alt="">'
    else:
        badge = tile.get("badge") or default_badge(tile["name"])
        sub = f'<div class="sub">{html.escape(tile["sub"])}</div>' if tile.get("sub") else ""
        icon = (f'<div class="badge" style="color:{color};font-size:{badge_size(badge)}px">'
                f"{html.escape(badge)}</div>{sub}")
    return (f'<div class="tile"><div class="icon">{icon}</div>'
            f'<div class="label">{html.escape(tile["name"])}</div></div>')


def render_html(data):
    out = [f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS % {'w': PAGE_WIDTH}}</style></head><body>",
           f"<h1>{html.escape(data['title'])}</h1>"]
    for sec in data["section"]:
        c = sec["color"]
        out.append(f'<div class="section"><div class="shead"><div class="num" style="background:{c}">{sec["number"]}</div>'
                   f'<h2>{html.escape(sec["title"])}</h2><div class="rule" style="background:{c}"></div></div><div class="cards">')
        for card in sec.get("card", []):
            tiles = "".join(render_tile(t, c) for t in card["tiles"])
            out.append(f'<div class="card" style="border-top-color:{c}"><h3 style="color:{c}">{html.escape(card["title"])}</h3>'
                       f'<div class="tiles">{tiles}</div></div>')
        out.append("</div></div>")
    out.append(f"<footer><span>{html.escape(data['footer_left'])}</span><span>{html.escape(data['footer_right'])}</span></footer>")
    out.append("<script>window.addEventListener('load',()=>document.body.setAttribute('data-h',"
               "Math.ceil(document.documentElement.scrollHeight)))</script></body></html>")
    return "".join(out)


def find_chromium():
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(name):
            return shutil.which(name)
    sys.exit("No Chromium or Chrome binary found on PATH.")


def screenshot(page, out):
    chrome = find_chromium()
    common = [chrome, "--headless", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=15000"]
    dom = subprocess.run(common + [f"--window-size={PAGE_WIDTH},1000", "--dump-dom", page.as_uri()],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r'data-h="(\d+)"', dom)
    if not m:
        sys.exit("Could not measure page height.")
    height = int(m[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(common + [f"--window-size={PAGE_WIDTH},{height}", f"--force-device-scale-factor={SCALE}",
                             f"--screenshot={out}", page.as_uri()], capture_output=True, check=True)
    return PAGE_WIDTH * SCALE, height * SCALE


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch-logos", action="store_true", help="download missing GitHub avatar logos")
    ap.add_argument("--check-only", action="store_true", help="validate and report coverage without rendering")
    ap.add_argument("--out", type=Path, default=BUILD / "inference-landscape.png")
    args = ap.parse_args()

    data = tomllib.loads((HERE / "landscape.toml").read_text(encoding="utf-8"))
    readme = parse_readme(ROOT / "README.md")
    if args.fetch_logos:
        fetch_logos(data)

    errors, warnings, covered = check(data, readme)
    tiles = sum(len(c["tiles"]) for s in data["section"] for c in s.get("card", []))
    print(f"{tiles} tiles, {len(readme[0])} README entries")
    uncovered = [f"  #{a}  {t}" for a, t in readme[3] if a not in covered and not a.startswith(("courses", "repositories", "talks"))]
    if uncovered:
        print(f"README subsections with no tile ({len(uncovered)}):")
        print("\n".join(uncovered))
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        sys.exit(1)
    if args.check_only:
        return

    BUILD.mkdir(exist_ok=True)
    page = BUILD / "landscape.html"
    page.write_text(render_html(data), encoding="utf-8")
    w, h = screenshot(page, args.out.resolve())
    print(f"wrote {args.out} ({w} x {h})")


if __name__ == "__main__":
    main()
