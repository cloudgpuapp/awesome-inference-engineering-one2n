# Contributing

Thanks for helping make this a useful, opinionated inference engineering resource.

## Additions

1. Search the README to avoid duplicate entries.
2. Pick the most precise section in README.md, or propose a new category via an issue.
3. Add one Markdown bullet: `- [Project](https://github.com/org/repo) - What it offers for inference engineers.`
   - Papers: `- [Title](https://arxiv.org/abs/xxxx.xxxxx) - **Paper, YYYY**. What it contributes.` Use the initial preprint or publication year.
   - Commercial books or courses: start the description with `**Paid book**.` or `**Paid course**.`
   - Archived or maintenance-mode projects kept for background: end the description with `Historical: <status>.`
   - Research implementations whose methods now ship in production engines: end the description with `Reference code: <where it ships>.`
   - Use ASCII characters only in entries and prose.
4. Explain the resource's inference relevance in your pull request.
5. Use direct, stable links; prefer upstream/source URLs.
6. Mark a project as archived, unmaintained or historical when appropriate.
7. Never present a vendor's performance claim as an independently validated result.
8. Test links and Markdown structure before submitting.

## Inclusion criteria

- Useful for model execution, inference architecture, optimized kernels, serving, hardware, observability, benchmarking, capacity or reliability.
- Real technical material: code, documentation, paper, reproducible experiment or high-quality talk.
- Distinct value beyond entries already listed.
- Clear scope and a short explanation.

## Exclusions

- Generic prompt collections, agent app directories, unrelated training tutorials and SEO-only articles.
- Bare marketing or unrepeatable speed claims.
- Duplicate forks unless they have meaningful divergence.
- Link dumps and affiliate content.

## Paper submissions

Include paper title, publication year, link, code if available, technical contribution, and relevant workloads.

## Benchmark submissions

For benchmark submissions, supply the accelerator, model revision, precision, serving engine and commit, input/output distributions, concurrency or offered QPS, cache state, methodology, date, and measurements.

## Landscape image

`assets/inference-landscape.png` is generated from `landscape/landscape.toml`. Do not edit the PNG directly.

A tile is optional. Add one only for a tool or concept that belongs on the overview; most README entries do not need one.

To add a tile, add one line to the right card in `landscape/landscape.toml`. Its `link` must exactly match the URL of a README entry, or a README section anchor such as `#kv-cache-basics`:

```toml
{ name = "XGrammar", logo = "gh", link = "https://github.com/mlc-ai/xgrammar" },
```

Use `logo = "gh"` for the repo owner's GitHub avatar, `logo = "gh:owner"` for another GitHub owner, or a file in `landscape/logos/`. Without a logo, set `badge` (short text) and optionally `sub` (vendor).

Commands (Python 3.11+; rendering also needs Chromium or Chrome):

```sh
python3 landscape/build.py --check-only     # validate tiles against README.md and list uncovered sections
python3 landscape/build.py --fetch-logos    # download missing GitHub avatars into landscape/logos/
python3 landscape/build.py                  # render a preview to landscape/build/ (not committed)
python3 landscape/build.py --out assets/inference-landscape.png   # maintainers: publish the image
```

Run `--check-only` before opening a pull request. Maintainers regenerate the published image so renders stay consistent.

## Review checklist

- [ ] The link works and the entry is not duplicated.
- [ ] Description is factual and concise.
- [ ] Inference relevance is apparent.
- [ ] Maintenance status is disclosed if necessary.
- [ ] Performance claims have context and provenance.
- [ ] If a tile was added, `python3 landscape/build.py --check-only` passes.

Maintainers may decline submissions to keep the list curated rather than exhaustive.

Run `python3 .github/scripts/check_internal_links.py` to check local files and section anchors.
