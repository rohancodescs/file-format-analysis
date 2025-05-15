# bench_single.py  (label-sanity version)  ───────────────────────────────
import sys, time, csv, psutil, numpy as np
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq, xgboost as xgb

FMT         = sys.argv[1]            # CSV / Parquet / HDF5
SAMPLE_ROWS = 1_000_000
NTHREADS    = 4
RESULTS_CSV = Path("benchmark_results.csv")

def downcast(df: pd.DataFrame):
    fcols = df.select_dtypes("float64").columns
    df[fcols] = df[fcols].astype(np.float32, copy=False)
    return df

def sampler(df):
    if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=0)
    return df

def load_csv(path="data.csv", chunksize=250_000):
    pieces, rows = [], 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        need = SAMPLE_ROWS - rows if SAMPLE_ROWS else len(chunk)
        frac = min(1.0, need / len(chunk))
        pieces.append(chunk.sample(frac=frac, random_state=0))
        rows += len(pieces[-1])
        if SAMPLE_ROWS and rows >= SAMPLE_ROWS:
            break
    return downcast(pd.concat(pieces, ignore_index=True))

def load_parquet(path="data_2gb.parquet"):
    return downcast(sampler(pq.read_table(path).to_pandas()))

def load_hdf5(path="data_2gb.h5"):
    return downcast(sampler(pd.read_hdf(path, "train")))

loaders = {"CSV": load_csv, "Parquet": load_parquet, "HDF5": load_hdf5}

# load & preprocess 
proc  = psutil.Process()
t0    = time.perf_counter()
df    = loaders[FMT]()
load_s = time.perf_counter() - t0
mem_gb = proc.memory_info().rss / 1024**3

# numeric feature matrix
num_cols = df.select_dtypes(include=["float32","float64","int8","int16","int32","int64"]).columns
num_cols = num_cols.drop(["target", "test"])
X = df[num_cols].astype(np.float32, copy=False)

# label: map any value not zero or 1 to 0  
y = df["target"].where(df["target"] == 1, 0).astype(np.int8, copy=False)

dtrain = xgb.DMatrix(X.values, label=y.values)

t1 = time.perf_counter()
xgb.train({"objective":"binary:logistic",
           "tree_method":"hist",
           "nthread": NTHREADS},
          dtrain, num_boost_round=50, verbose_eval=False)
train_s = time.perf_counter() - t1

row = {"format": FMT, "rows": len(y),
       "load_sec": round(load_s,2),
       "train_sec": round(train_s,2),
       "peak_ram_gb": round(mem_gb,2)}
print(row)

# log
write_header = not RESULTS_CSV.exists()
with RESULTS_CSV.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if write_header:
        w.writeheader()
    w.writerow(row)
# ────────────────────────────────────────────────────────────────────────
