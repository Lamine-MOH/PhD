# AGENTS.md

## What this is

PhD research repo for retinal image analysis (diabetic retinopathy grading). Early stage — single file `src/Data.py` with dataset download/preparation/loading utilities.

## Datasets

Four retinal image datasets: **Aptos**, **IDRiD**, **DDR**, **Messidor-2**. Downloaded via `kagglehub` (Aptos, DDR, Messidor-2) or `gdown` from Google Drive (IDRiD). Kaggle auth must be configured for kagglehub downloads.

## Key dependencies

Python: `torch`, `torchvision` (PIL, sklearn, pandas, matplotlib, numpy, kagglehub, gdown). Install with `pip install -r requirements.txt`.


