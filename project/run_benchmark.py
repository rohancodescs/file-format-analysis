# run_benchmark.py ──────────────────────────────────────────────────────
import sys, time, csv, psutil, numpy as np
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq, xgboost as xgb
from tqdm import tqdm    
import os           
import pyarrow.parquet as pq
import pyarrow as pa
# ----------------------------- settings --------------------------------
FMT            = sys.argv[1]               # CSV  Parquet  HDF5
RESULTS_CSV    = Path("benchmark_results.csv")
XGB_THREADS    = 4
POST_SAMPLE_N  = 1_000_000   # None = keep all rows *after* load/parse
# -----------------------------------------------------------------------

def downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    fcols = df.select_dtypes("float64").columns
    df[fcols] = df[fcols].astype(np.float32, copy=False)
    return df

# def load_csv(path="data.csv") -> pd.DataFrame:
#     df = pd.read_csv(path, low_memory=False)
#     return downcast_float32(df)
# def load_parquet(path="data_2gb.parquet") -> pd.DataFrame:
#     # read *all* row-groups with Arrow → pandas
#     df = pq.read_table(path).to_pandas()
#     return downcast_float32(df)

# def load_hdf5(path="data_2gb.h5") -> pd.DataFrame:
#     df = pd.read_hdf(path, "train")
#     return downcast_float32(df)
def load_csv(path="data.csv",
             chunksize=250_000,
             concat_every=8):           # concat after N chunks → no big freeze
    total_bytes = os.path.getsize(path)
    parts, big_parts, chunks = [], [], 0

    with tqdm(total=total_bytes, unit="B", unit_scale=True,
              desc="CSV read") as bar:
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            chunk  = downcast_float32(chunk)
            parts.append(chunk)
            chunks += 1

            # update progress bar by estimated bytes of this chunk in memory
            bar.update(chunk.memory_usage(index=False).sum())

            # periodically concatenate to avoid one giant concat at the end
            if chunks % concat_every == 0:
                big_parts.append(pd.concat(parts, ignore_index=True))
                parts = []              # reset small buffer

        # concat any leftovers
        if parts:
            big_parts.append(pd.concat(parts, ignore_index=True))

    return pd.concat(big_parts, ignore_index=True)

# ─────────────────── 2) Parquet row-group progress  ────────────────────
def load_parquet(path="data_2gb.parquet"):
    pq_file = pq.ParquetFile(path)
    n_rg    = pq_file.num_row_groups
    tables  = []
    with tqdm(total=n_rg, desc="Parquet row-groups") as bar:
        for i in range(n_rg):
            tables.append(pq_file.read_row_group(i))
            bar.update(1)
    df = downcast_float32(pa.concat_tables(tables).to_pandas())   # ← use pa.concat_tables
    return df

# ──────────────────── 3) HDF5 chunk progress  ──────────────────────────
def load_hdf5(path="data_2gb.h5", chunksize=250_000):
    store  = pd.HDFStore(path, "r")
    nrows  = store.get_storer("train").nrows
    parts  = []
    with tqdm(total=nrows, desc="HDF5 rows") as bar:
        for chunk in store.select("train", chunksize=chunksize):
            parts.append(downcast_float32(chunk))
            bar.update(len(chunk))
    store.close()
    return pd.concat(parts, ignore_index=True)


LOADERS = {"CSV": load_csv, "Parquet": load_parquet, "HDF5": load_hdf5}

# --------------------------- benchmark ---------------------------------
proc   = psutil.Process()
t0     = time.perf_counter()
df     = LOADERS[FMT]()            # FULL read + dtype cast
load_s = time.perf_counter() - t0
mem_gb = proc.memory_info().rss / 1024**3

# optional post-load sample (does NOT count in load_s)
if POST_SAMPLE_N and len(df) > POST_SAMPLE_N:
    df = df.sample(n=POST_SAMPLE_N, random_state=0)

# build X / y
num_cols = df.select_dtypes(include=["number"]).columns.drop(["target", "test"])
X = df[num_cols].to_numpy(dtype=np.float32, copy=False)
# y = df["target"].fillna(0).astype(np.int8).to_numpy(copy=False) #enable for parquet / csv
y = df["target"].clip(lower=0).astype(np.int8, copy=False) #enable for hdf5


dtrain = xgb.DMatrix(X, label=y)
t1 = time.perf_counter()
xgb.train({"objective": "binary:logistic",
           "tree_method": "hist",
           "nthread": XGB_THREADS},
          dtrain, num_boost_round=50, verbose_eval=False)
train_s = time.perf_counter() - t1

row = {"format": FMT,
       "rows": len(y),
       "load_sec": round(load_s, 2),
       "train_sec": round(train_s, 2),
       "peak_ram_gb": round(mem_gb, 2)}
print(row)

# ----------------------- append to CSV log -----------------------------
write_header = not RESULTS_CSV.exists()
with RESULTS_CSV.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if write_header:
        w.writeheader()
    w.writerow(row)
# -----------------------------------------------------------------------
