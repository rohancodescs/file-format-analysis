#imports
import xgboost as xgb
from tqdm import tqdm
import csv, os
import time
import psutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

#benchmark settings
RESULTS_FILE = Path("benchmark_results.csv")  # where we log runs
XGB_THREADS = 4 # CPU threads for XGBoost
POST_SAMPLE_N = 1_000_000 # sample after load (None -> all)

# converting every float64 column to float32 in-place to save RAM
def downcast(df: pd.DataFrame) -> pd.DataFrame:
    f64 = df.select_dtypes("float64").columns #name of every float64 column 
    df[f64] = df[f64].astype(np.float32, copy=False) #in-place cast, no copy
    return df #returning lighter df

#loaders for each file format

#csv loader with progress bar
def load_csv(path="data.csv", chunksize=250_000, concat_every=8):
    total = os.path.getsize(path) #file size in bytes
    buf = [] #staging the buffers
    big_parts = [] #big parts buffer
    n_chunks = 0 #chunk counter
    #tqdm shows a live counter of the bytes while reading the text file
    with tqdm(total=total, unit="B", unit_scale=True, desc="CSV read") as bar: 
        # pandas iterator streams 'chunksize' rows at a time
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            buf.append(downcast(chunk))
            n_chunks += 1 #casting and staging chunk
            bar.update(chunk.memory_usage(index=False).sum()) #advance bar
            # every "concat_every" chunks, concatenate and append to big_parts
            if n_chunks % concat_every == 0:
                big_parts.append(pd.concat(buf, ignore_index=True))
                buf = [] #resets small buffer
        #flushing leftover chunks
        if buf:
            big_parts.append(pd.concat(buf, ignore_index=True))
    #final df concats N big parts
    return pd.concat(big_parts, ignore_index=True)

# Parquet loader
def load_parquet(path="data.parquet"):
    pqf = pq.ParquetFile(path) #open once, zero copy i/o
    tables = [] # arrow table buffer
    with tqdm(total=pqf.num_row_groups, desc="Parquet RG") as bar: 
        #iterating through the row groups
        for i in range(pqf.num_row_groups):
            tables.append(pqf.read_row_group(i)) #read row group -> arrow table
            bar.update(1) #progress tick
    #concat arrow tables, convert to pandas, downcast floats from float64 -> float32
    return downcast(pa.concat_tables(tables).to_pandas())

# HDF5 loader (chunked read)
def load_hdf5(path="data.h5", chunksize=250_000):
    store = pd.HDFStore(path, "r") #open HDF5 in read-only mode
    nrows = store.get_storer("train").nrows #total rows in tqdm
    parts = [] #collected dfs
    with tqdm(total=nrows, desc="HDF5 rows") as bar: 
        # per select, pytables streams a row of size "chunksize"
        for chunk in store.select("train", chunksize=chunksize): #
            parts.append(downcast(chunk))  #casting and staging
            bar.update(len(chunk)) # progress rows read
    store.close() #close file handle
    return pd.concat(parts, ignore_index=True) #stitching chunks together

LOADERS = {"CSV": load_csv, "Parquet": load_parquet, "HDF5": load_hdf5}
#loading the file, sample rows, training XGboost, and logging timings
def run_benchmark(fmt: str, path: str):
    #measuring ram before and after load phase
    proc = psutil.Process() # for RAM measurement
    t0 = time.perf_counter() #starting timer
    df = LOADERS[fmt](path) # full ingest + float32 cast
    load_sec = time.perf_counter() - t0 #elapsed time to parse / load
    peak_ram = proc.memory_info().rss / 1024**3 #resident set size (RAM used) in GB

    # down-sample so every format trains on equal rows (1M)
    if POST_SAMPLE_N and len(df) > POST_SAMPLE_N:
        df = df.sample(n=POST_SAMPLE_N, random_state=0)

    # building feature matrix and label vector
    num_cols = df.select_dtypes("number").columns.drop(["target", "test"])
    X = df[num_cols].to_numpy(dtype=np.float32, copy=False)
    if fmt == "HDF5":
        y = df["target"].clip(lower=0).astype(np.int8, copy=False) #enable for hdf5
    else:
        y = df["target"].fillna(0).astype(np.int8).to_numpy(copy=False) #enable for csv/parquet

    # XGBoost training
    dtrain = xgb.DMatrix(X, label=y) 
    t1 = time.perf_counter()
    xgb.train({"objective": "binary:logistic", #0/1 classification
               "tree_method": "hist", #fast CPU histogram algorithm
               "nthread": XGB_THREADS}, # limit CPU threads
              dtrain, num_boost_round=50, verbose_eval=False) #50 boosting rounds
    train_sec = time.perf_counter() - t1 

    # log and display results
    result = {"format": fmt, "rows": len(y),"load_sec": round(load_sec, 2), "train_sec": round(train_sec, 2), "peak_ram_gb": round(peak_ram, 2)}
    print(f"{fmt}: {len(y):,} rows, load: {load_sec:.2f}s, train: {train_sec:.2f}s, peak RAM: {peak_ram:.2f}GB")

    # append row to CSV log
    write_header = not RESULTS_FILE.exists() #adding header only once
    with RESULTS_FILE.open("a", newline="") as f: #
        w = csv.DictWriter(f, fieldnames=result.keys())
        if write_header: w.writeheader()
        w.writerow(result)
# running the benchmark on each stored file 
run_benchmark("CSV", "data.csv")
run_benchmark("Parquet", "data.parquet")
run_benchmark("HDF5", "data.h5")