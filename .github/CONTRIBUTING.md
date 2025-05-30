# Contributing

Before the first commit:

```bash
pip install -e .[dev,scripts]
pre-commit install
pre-commit run --all-files
```

Run `pre-commit` after staging, and before committing. Make sure all the tests pass (By running `pytest`). Note that the `pytest` hook of `pre-commit` only runs the tests in the `testing/` folder. To run all the tests, which take longer, run `pytest` manually.

The codebase uses:
- `black` formatter for code style
- `flake8` for linting
- `pyright` for static type checking

The pre-commit hooks ensure code quality and type safety across the project. The pyright configuration includes all project directories including tutorials/examples and testing.

## Recommended Editor Setup (VS Code / Cursor)

For the best development experience, we recommend using VS Code or Cursor with the following extensions:

- Python: Core Python support.
- Black Formatter: Integrates the `black` code formatter.
- isort: Integrates the `isort` import sorter.
- Pyright: Provides static type checking.

**Important:** This project uses `pyright` for static type checking. To ensure consistency with the pre-commit hooks and avoid conflicting diagnostics, please install the **Pyright** extension and **disable** the **Pylance** extension in your VS Code/Cursor workspace settings.

## Making Documentation Changes Locally

To make the docs locally:

```bash
cd docs
make html
open build/html/index.html
``` 