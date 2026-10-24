# Contributing

Thanks for helping make this a useful, opinionated inference engineering resource.

## Additions

1. Search the README to avoid duplicate entries.
2. Pick the most precise section in README.md, or propose a new category via an issue.
3. Add one Markdown bullet: `- [Project](https://github.com/org/repo) - What it offers for inference engineers.`
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

## Review checklist

- [ ] The link works and the entry is not duplicated.
- [ ] Description is factual and concise.
- [ ] Inference relevance is apparent.
- [ ] Maintenance status is disclosed if necessary.
- [ ] Performance claims have context and provenance.

Maintainers may decline submissions to keep the list curated rather than exhaustive.

Run `python .github/scripts/check_internal_links.py` to check local files and section anchors.
