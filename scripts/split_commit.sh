#!/usr/bin/env bash
set -euo pipefail

# scripts/split_commit.sh
# Automate a sequence of focused commits. Each step lists files to add
# and a commit message. Missing files are skipped; steps with no added
# files are reported and not committed.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

declare -a steps

# Format: files|commit message
steps+=("README.md requirements.txt|docs: update README setup and add pinned requirements")
steps+=("src/config.py|chore: add central config (paths, hyperparams, class list)")
steps+=("src/prepare_data.py|feat(data): add dataset splitting and processed layout script")
steps+=("src/data_loader.py|feat(data): add tf.data pipeline and augmentation")
steps+=("src/build_model.py|feat(model): MobileNetV2 transfer-learning model and fine-tune helper")
steps+=("src/train.py|train: two-stage training script with checkpointing and plotting")
steps+=("src/evaluate.py|test: evaluation script to compute precision/recall/F1 and confusion matrix")
steps+=("src/convert_to_tflite.py|build: SavedModel export and TFLite conversion with quantization check")
steps+=("src/dashboard.py src/db.py|ui: Streamlit dashboard + SQLite persistence for inspection history")
steps+=("notebooks/exploration.ipynb results/metrics_report.txt results/class_metrics.json|docs: add exploration notebook and evaluation artifacts (metrics manifest)")

echo "Running split commit script from $ROOT_DIR"

for step in "${steps[@]}"; do
  IFS='|' read -r files msg <<< "$step"
  echo "\n-> Commit: $msg"

  added_any=false
  for f in $files; do
    # Support globs; expand and check
    for match in $f; do
      if [ -e "$match" ]; then
        git add "$match"
        added_any=true
        echo "  added: $match"
      fi
    done
  done

  if $added_any; then
    git commit -m "$msg" || { echo "Commit failed for: $msg"; exit 1; }
  else
    echo "  (no existing files to add for this step)"
  fi
done

echo "\nAll steps processed. To push the branch to origin:"
echo "  git push -u origin HEAD"
