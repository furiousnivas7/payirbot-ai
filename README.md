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
- If `pip install` fails due to network or proxy restrictions, use a local
	wheel cache or configure `HTTPS_PROXY` / `HTTP_PROXY` before retrying.

## Contributing
- Create feature branches, keep commits focused (this repo prefers small
	logical commits for history clarity), and open a PR when ready.

## License
Pick a license and add a `LICENSE` file (MIT is common for research prototypes).

