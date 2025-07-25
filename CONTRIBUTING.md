# Contribution Guide

## Processus de mise en production

The main branch is the trunk and features branches are squashed merge after a successful Pull request.

Contributer steps:
- [X] make sure the tests pass locally
- [X] use `ruff format` and `ruff check` to make sure the formatting is correct
- [X] `CHANGELOG.md` documents the changes, references the PR, the date and the new version number in `./src/ephysatlas/__init__.py`
- [X] create a PR from your feature branch to main

Reviewer steps:
- [ ] the CI passes
- [ ] squash-merge upon a successful review
- [ ] create tag corresponding to the version number `X.X.X` on the `main` branch
