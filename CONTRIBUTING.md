# Contributing to PayirBot Training

Thank you for contributing to PayirBot Training! This repository is focused on creating a clear, reproducible workflow for plant disease classification and model deployment.

## Getting started

1. Fork the repository.
2. Create a new feature branch:

```bash
git checkout -b feat/your-feature-name
```

3. Make your changes.
4. Run the repository checks:

```bash
make test
```

5. Commit with a clear message.
6. Push your branch and open a pull request.

## Branch and commit style

- Use feature branches for new work.
- Keep commits small and logical.
- Use conventional-style messages when possible, e.g.:
  - `feat: add dashboard image upload`
  - `fix: handle missing dataset path`
  - `docs: improve setup instructions`
  - `chore: update dependencies`

## Testing

This repository currently checks Python source syntax and import consistency via:

```bash
make test
```

If you add new scripts, include them in the Makefile and CI workflow.

## Offline dependency support

If the contributor is working in an environment without direct internet access, use:

```bash
make wheelhouse
```

Copy the generated `wheelhouse/` directory to the offline machine and install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-index --find-links=wheelhouse -r requirements.txt
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License in this repository.
