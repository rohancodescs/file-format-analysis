# MSML-605 Final Project

**Investigating the Impact of Storage Formats on End‑to‑End ML Performance**
Rohan Bhatt • Shubhang Srikoti

---

## 1 · Overview

This project benchmarks three storage formats—CSV, Snappy‑Parquet, and Blosc‑Zstd HDF5—on their impact to data‑load latency, peak memory, and XGBoost training time. All experiments use the **American Express Default Prediction** dataset (≈17 M rows, 193 columns).

| File                    | Purpose                                                                         |
| ----------------------- | ---------------------------------------------------------------------------       |
| `project2.ipynb`        | **Main notebook**: converts formats, runs benchmarks, creates plots.                            |
| `run_benchmark.py and bench_single.py (OLD)`       | CLI helper that executes one benchmark head‑less (development aid).         |
| `proj.ipynb`            | Early prototype exploring HDF5 compression levels.                                                      |
| `benchmark_results.csv` | Auto‑appended log of every benchmark run.                                   |
| *data files*            | Generated locally (git‑ignored): `data.parquet`, `data.csv`, `data.h5`. |

---

## 2 · Quick‑Start (pip)

```bash
# clone the repository
git clone https://github.com/rohancodescs/file-format-analysis

# create a virtual environment
python -m venv .venv
source .venv/bin/activate 

# download the dataset (≈12 GB) with Kaggle CLI (first cell in jupyter notebook, instructions below if you want to use CLI)
pip install kaggle   # if you don't already have it
kaggle competitions download -c amex-default-prediction -f train_data.parquet
mv train_data.parquet data.parquet
```

Open **`project2.ipynb`** in Jupyter and run cells top‑to‑bottom. The notebook will:

1. Convert `data.parquet` → `data.csv` and `data_hex.h5`.
2. Load each format with tqdm progress bars.
3. Train XGBoost (hist, 50 rounds) on a 1 M‑row sample.
4. Append timings to `benchmark_results.csv` and plot comparative charts.

> **Tip (macOS)** – prevent sleep during long conversions:
> `caffeinate -i python project2.ipynb`

---

## 3 · Project Dependencies

All packages are pinned in **`requirements.txt`**.

```
numpy
pandas
pyarrow
tables
xgboost
psutil
tqdm
matplotlib
jupyterlab   # notebook interface
```

Versions latest as of May 2025 (tested on Python 3.9.19 / macOS 14.4.1).

---

## 4 · Reproduce Benchmarks via CLI (optional)

```bash
python bench_single.py CSV
python bench_single.py Parquet
python bench_single.py HDF5
```

Each command prints a JSON result and appends the same row to `benchmark_results.csv`.


## 5 · Re‑Creating Artefacts

| Step           | Command (inside notebook)    | Approx time *M4 Pro CPU* |
| -------------- | ---------------------------- | -------------------- |
| Parquet → CSV  | cell “Convert to CSV”        | 25 min               |
| Parquet → HDF5 | cell “Convert to HDF5 (hex)” | 90+ min              |
| Benchmarks     | cell “Run benchmarks”        | 10-15 min              |

---
