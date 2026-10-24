#!/usr/bin/env python3
"""Validate local Markdown files and fragments using only the standard library."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r'\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[\"\'][^\n]*?[\"\'])?\s*\)')

def prose(text):
    return re.sub(r'(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$', '', text)

def anchors(text):
    ids = re.findall(r'<a\s+(?:id|name)=[\"\']([^\"\']+)[\"\']', text)
    counts = {}
    for title in re.findall(r'^#{1,6}\s+(.+?)(?:\s+#+)?\s*$', prose(text), re.M):
        title = re.sub(r'<[^>]*>', '', title).lower().replace(' ', '-')
        base = ''.join(c for c in title if c in '-_' or unicodedata.category(c)[0] in 'LN' or unicodedata.category(c) == 'Mn')
        n = counts.get(base, 0)
        counts[base] = n + 1
        ids.append(base + (f'-{n}' if n else ''))
    return set(ids)

errors = []
checked = 0
for file in sorted(ROOT.rglob('*.md')):
    text = file.read_text(encoding='utf-8')
    explicit = re.findall(r'<a\s+id="([^"]+)"', text)
    if len(explicit) != len(set(explicit)):
        errors.append(f'{file.relative_to(ROOT)}: duplicate explicit anchor')
    for match in LINK.finditer(prose(text)):
        target = match[1].strip('<>')
        parts = urlsplit(target)
        if parts.scheme or parts.netloc:
            continue
        checked += 1
        destination = (file.parent / unquote(parts.path)).resolve() if parts.path else file
        if not destination.is_file():
            errors.append(f'{file.relative_to(ROOT)}: missing file: {target}')
        elif parts.fragment and destination.suffix.lower() == '.md':
            if unquote(parts.fragment) not in anchors(destination.read_text(encoding='utf-8')):
                errors.append(f'{file.relative_to(ROOT)}: missing section: {target}')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print(f'All {checked} internal Markdown links resolve to existing files and sections.')
