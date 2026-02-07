# Contributing to NutriBin-MachineLearning

Thank you for your interest in contributing! We welcome improvements, bug reports, and new ideas — your help makes this project better.

## How to contribute

- Discuss big changes or feature ideas by opening an issue first.
- For bug reports, please include steps to reproduce, environment details, and any relevant logs or screenshots.
- When you're ready to contribute code, open a Pull Request (PR) with a clear title and description and link the related issue if one exists.

## Development setup

Recommended: use a virtual environment and install project dependencies.

```bash
python -m venv .venv
.venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

Run tests and linters where available (replace with repo-specific commands if present):

```bash
pytest
flake8
```

## Branching & PR guidelines

- Create a feature branch from `main`: `feature/short-description`.
- Keep commits focused and use clear commit messages.
- Add tests for new behavior where applicable.
- Ensure changes follow the repository style (PEP 8 for Python).

## Code style

- Follow PEP 8 and idiomatic Python.
- We recommend using `black` and `isort` for formatting and import order.

## Licensing and CLA

By contributing you agree that your contributions will be licensed under the project's existing license.

## Code of conduct

Please follow the project's [Code of Conduct](CODE_OF_CONDUCT.md). Respectful, inclusive behavior is expected.

Thanks again — contributions are appreciated!

## Pull Request checklist & template

Before opening a Pull Request (PR), please make sure your changes meet the checklist below. This helps reviewers and speeds up merging.

- [ ] I opened an issue describing the change (if the change is more than a small fix).
- [ ] My branch is up-to-date with `main`.
- [ ] My commits are small, atomic, and have clear messages.
- [ ] I added tests for new functionality or updated existing tests.
- [ ] I ran the test suite locally and all tests pass.
- [ ] I updated documentation or README if applicable.
- [ ] I followed the repository's code style and formatting guidelines.

Suggested PR description template (copy into the PR description):

```
Title: Short, descriptive title (e.g. Fix model training loss logging)

Description:
- What changed and why (one or two short paragraphs).
- Which issue this PR addresses (e.g. "Fixes #123") or reference to discussion.

Testing:
- How you tested these changes (commands, small summary of results).

Checklist:
- [ ] Tests added or updated
- [ ] Documentation updated
- [ ] Ready for review

Notes for reviewers:
- Anything specific the reviewer should look for, or known limitations.
```

Thanks — a clear PR description and checklist speeds up review and improves project quality.
