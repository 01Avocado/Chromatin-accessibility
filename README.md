# Chromatin accessibility (EpiBERT-style)

This repo contains preprocessing and training code for predicting chromatin accessibility from DNA sequence and functional genomics tracks.

## Repo layout

- `scripts/preprocess/`: preprocessing pipeline (BED → sequences/labels/features)
- `scripts/train/`: training + evaluation code
- `docs/`: project notes/documentation
- `data/`: datasets (ignored by git)
- `results/`: figures/metrics (ignored by git)
- `models/`: checkpoints (ignored by git)

## Running

Install deps:

```bash
pip install -r requirements.txt
```

Train (example):

```bash
python scripts/train/train_epibert.py --use_normalized_func --use_normalized_labels
```

Evaluate:

```bash
python scripts/train/evaluate_epibert.py --checkpoint_path models/pretrained/checkpoint_latest.pt
```

## Notes

- Large/generated files are intentionally ignored via `.gitignore` (data, results, checkpoints, PDFs/images).
