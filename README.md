# PayirBot Training

**Repository Name:** `payirbot-training`

**Description:** Transfer learning pipeline and TensorFlow Lite deployment for
plant disease classification focused on the PayirBot field inspection robot.

PayirBot Training is a plant disease classification repository that uses
MobileNetV2 transfer learning to train, evaluate, and deploy a crop inspection
model for the PayirBot autonomous field robot.

This repository contains data preparation, model training, evaluation,
TensorFlow Lite conversion, and a Streamlit dashboard used to simulate
on-robot inference and inspection history.

## Quick start

1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install pinned dependencies

```bash
pip install -r requirements.txt
```

Alternatively, use the provided helper commands:

```bash
make venv
make install
```

If you are behind a proxy, use:

```bash
make install-proxy PROXY=http://<proxy-host>:<proxy-port>
```

If the proxy uses self-signed certificates, you may also need to set:

```bash
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"
```

If you need offline setup, create a wheelhouse from a machine with internet access:

```bash
python3 -m venv .wheelenv
. .wheelenv/bin/activate
python -m pip install --upgrade pip
python -m pip wheel -r requirements.txt -w wheelhouse
```

Then copy the `wheelhouse/` directory to the target machine and install from it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-index --find-links=wheelhouse -r requirements.txt
```

3. Download the dataset (Kaggle)

```bash
cd data/raw
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip
cd ../..
```

4. Prepare processed data

```bash
python src/prepare_data.py
```

5. Train (two-stage transfer learning)

```bash
caffeinate -i python src/train.py
```

6. Evaluate (writes `results/class_metrics.json` used by the dashboard)

```bash
python src/evaluate.py
```

7. Convert to TensorFlow Lite

```bash
python src/convert_to_tflite.py
```

8. Run the Streamlit dashboard

```bash
streamlit run src/dashboard.py
```

## Project layout

Key folders and files:
- `data/` — raw and processed datasets (processed created by `prepare_data.py`)
- `src/` — scripts (config, data pipeline, model, train/evaluate, dashboard)
- `models/` — model artifacts (ignored by default; use Releases or LFS)
- `results/` — evaluation outputs (confusion matrix, metrics)
- `dashboard_data/` — local SQLite DB and captured images (ignored)

## Notes
- Large files (datasets, model binaries, results) are excluded via `.gitignore`.
	If you want to version small result artifacts like `results/class_metrics.json`,
	whitelist them in `.gitignore` instead of committing whole folders.
- For macOS Apple Silicon, prefer `tensorflow-macos` + `tensorflow-metal`.
- If `pip install` fails due to network or proxy restrictions, configure
	`HTTPS_PROXY` / `HTTP_PROXY` or use a local wheel cache. The repository can
	also be prepared via `make venv` and `make install` once connectivity is fixed.
- If you are behind a proxy, run `make install-proxy PROXY=http://<proxy-host>:<proxy-port>`.
- For an offline install, generate a `wheelhouse/` directory from a connected
	machine and install using `--no-index --find-links=wheelhouse`.

## Developer workflow
- Use `make venv` to bootstrap a local Python environment.
- Use `make install` to install dependencies.
- Use `make wheelhouse` to generate offline wheel files in `wheelhouse/`.
- Use `make test` to run repository sanity checks and import tests.

## Contributing
- Create feature branches and keep commits small and focused.
- Use clear commit messages that describe intent, e.g. `feat: add dashboard UI` or
  `fix: resolve dataset path handling`.
- Open a pull request when ready and document your changes in the PR description.

## License
This repository is released under the MIT License. See `LICENSE` for details.

