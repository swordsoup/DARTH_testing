"""
build_glove_twitter.py

Builds a single-embedding-space GLOVE100 dataset for DARTH from the
ann-benchmarks glove-100-angular.hdf5 file (GloVe Twitter 27B, 100d).

Why this exists
---------------
The stock DARTH pipeline (manage_glove.py) mixes two GloVe releases: base
vectors from the ann-benchmarks HDF5 (Twitter 27B, 1.19M vocab) and
learn/query vectors from glove.6B.100d.txt (Wikipedia+Gigaword, 400K vocab).
Those are different embedding spaces -- the axes do not correspond, so a
word's neighbours in one space are unrelated to its neighbours in the other.
Recall would still be well-defined (groundtruth and search use the same base
set and metric) but query "hardness" and "topic" would be artifacts of the
space mismatch, which is exactly what we want to study.

This script builds everything from the HDF5 alone, and recovers the word for
every vector by exact-matching against glove.twitter.27B.100d.txt.

Splits
------
    base            1,072,514   hdf5 train, remainder
    learn             100,000   hdf5 train, held out
    validation         10,000   hdf5 train, held out
    query              10,000   hdf5 test, unchanged
    query_english       1,000   frequency-head ASCII tokens, held out

Two query sets on purpose. The hdf5 test set is a uniform sample of a 1.19M
Twitter vocabulary, so it is almost entirely long-tail tokens (foreign script,
hashtag fragments, misspellings). query_english is drawn from the frequency
head, so it is common ASCII tokens -- dictionary words and netspeak alike.
Running both against the same index lets you compare recall between common and
rare tokens directly.

Note that frequency is confounded with position in the space: high-frequency
tokens sit in dense, well-connected regions and are generally easier to search.
So a recall gap between the two query sets reflects frequency and density, not
language or topic on its own. Every query set ships with a frequency-rank file
so you can control for this in the analysis.

Held-out matters: any query vector left in the base set is its own nearest
neighbour at distance zero, which inflates its recall.

Outputs use the filenames VectorDataLoader.cpp hardcodes for GLOVE100
(lines 181-196), plus word and frequency-rank maps that are not part of DARTH.

Prereqs
-------
    - glove-100-angular.hdf5           in MISC
    - glove.twitter.27B.100d.txt       in MISC
    - hdf5_word_map.pkl                in MISC  (from recover_words.py)

After running, patch VectorDataLoader.cpp line 181 to the base filename this
script prints, then rebuild. The script prints the exact commands at the end.

Run under tmux. Groundtruth is ~121K queries x 1.07M base at k=1000.
"""

import numpy as np
import faiss
import h5py
import pickle
import json
import os
import time

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MISC = "/mnt/data/darth-datasets/raw/GLOVE100/misc"
OUT = "/mnt/data/darth-datasets/processed/GLOVE100"

SEED = 42

LEARN_N = 100_000
VALIDATION_N = 10_000
ENGLISH_QUERY_N = 1_000

# English queries are drawn from the first N lines of the Twitter text file,
# which is sorted by descending corpus frequency. 50K keeps us to tokens that
# are genuinely common. Raise this if the candidate count comes up short.
FREQ_HEAD = 50_000

K = 1000        # groundtruth depth, matches every other DARTH dataset
D = 100         # GloVe Twitter 100d

os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------
# File writers -- copied verbatim from organize_datasets.py so the binary
# layout is byte-identical to the GIST1M files that already work.
#
# These loop per-vector in Python. For the 1.07M-row base file that is slow
# (expect several minutes). Left as-is deliberately: this is the function that
# produced the working GIST1M dataset, and correctness beats speed here.
# ----------------------------------------------------------------------------

def write_ivecs(file_name, indices):
    with open(file_name, 'wb') as f:
        for vec in indices:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.int32).tobytes())


def write_fvecs(file_name, vectors):
    with open(file_name, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.float32).tobytes())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------

log("loading word map...")
with open(f"{MISC}/hdf5_word_map.pkl", "rb") as f:
    wm = pickle.load(f)
words = wm["words"]              # .txt line number -> word
train_idx = wm["train_idx"]      # hdf5 train row  -> .txt line number
test_idx = wm["test_idx"]        # hdf5 test  row  -> .txt line number

assert all(i is not None for i in train_idx), "unmatched rows in train_idx"
assert all(i is not None for i in test_idx), "unmatched rows in test_idx"
log(f"  vocab={len(words)} train={len(train_idx)} test={len(test_idx)}")

# The .txt file is sorted by descending corpus frequency, so its line number is
# a direct frequency rank. 0 = most common token in the corpus. This is a key
# covariate for the analysis, so keep it as arrays.
train_idx = np.asarray(train_idx, dtype=np.int64)
test_idx = np.asarray(test_idx, dtype=np.int64)

log("loading hdf5 vectors...")
with h5py.File(f"{MISC}/glove-100-angular.hdf5", "r") as f:
    train = f["train"][:].astype(np.float32)
    test = f["test"][:].astype(np.float32)
assert train.shape[1] == D and test.shape[1] == D
log(f"  train={train.shape} test={test.shape}")

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Reserve English queries FIRST
#
# Ordering matters. If you split base/learn/validation first and then filter
# the leftovers for English tokens, there are almost none left -- base is ~91%
# of train, so it swallows nearly the whole frequency head. Pick the English
# set up front, then exclude it from everything else.
# ----------------------------------------------------------------------------

log("selecting English query set...")

# .txt line number -> hdf5 train row, so we can go from frequency rank to vector
txtline_to_trainrow = {int(t): r for r, t in enumerate(train_idx)}

english_rows = []
for txt_line in range(min(FREQ_HEAD, len(words))):
    w = words[txt_line]
    # isascii() first: isalpha() is True for plenty of non-Latin scripts.
    if not (w.isascii() and w.isalpha() and len(w) > 2):
        continue
    row = txtline_to_trainrow.get(txt_line)
    if row is not None:
        english_rows.append(row)

log(f"  candidates in top {FREQ_HEAD} lines: {len(english_rows)}")
if len(english_rows) < ENGLISH_QUERY_N:
    raise SystemExit(
        f"only {len(english_rows)} English candidates, need {ENGLISH_QUERY_N}. "
        f"raise FREQ_HEAD and rerun."
    )

english_rows = np.array(english_rows, dtype=np.int64)
rng.shuffle(english_rows)
english_rows = english_rows[:ENGLISH_QUERY_N]
log(f"  selected {len(english_rows)}")


# ----------------------------------------------------------------------------
# Split the remaining train rows
#
# Mask instead of a Python loop over 1.18M rows. The three slices below are
# disjoint by construction, so no overlap assertions are needed.
# ----------------------------------------------------------------------------

log("splitting train...")
mask = np.ones(len(train), dtype=bool)
mask[english_rows] = False
remaining = np.flatnonzero(mask)
rng.shuffle(remaining)

learn_rows = remaining[:LEARN_N]
validation_rows = remaining[LEARN_N:LEARN_N + VALIDATION_N]
base_rows = remaining[LEARN_N + VALIDATION_N:]

BASE_N = len(base_rows)
BASE_FILENAME = f"base.{BASE_N}.fvecs"

log(f"  base={BASE_N} learn={len(learn_rows)} "
    f"validation={len(validation_rows)} query={len(test)} "
    f"query_english={len(english_rows)}")
log(f"  base filename: {BASE_FILENAME}")


# ----------------------------------------------------------------------------
# Materialise vectors
# ----------------------------------------------------------------------------

base_vecs = train[base_rows]
learn_vecs = train[learn_rows]
validation_vecs = train[validation_rows]
query_vecs = test
english_vecs = train[english_rows]

log("writing vectors...")
write_fvecs(f"{OUT}/{BASE_FILENAME}", base_vecs)
log(f"  {BASE_FILENAME}")
write_fvecs(f"{OUT}/learn.100K.fvecs", learn_vecs)
log("  learn.100K.fvecs")
write_fvecs(f"{OUT}/validation.10K.fvecs", validation_vecs)
log("  validation.10K.fvecs")
write_fvecs(f"{OUT}/query.10K.fvecs", query_vecs)
log("  query.10K.fvecs")
write_fvecs(f"{OUT}/query.english.1K.fvecs", english_vecs)
log("  query.english.1K.fvecs")


# ----------------------------------------------------------------------------
# Word and frequency-rank maps
#
# Not part of DARTH -- this is the join key for the recall analysis. Row i of
# each .fvecs file corresponds to entry i of the matching .words.json and
# .freqrank.json here.
#
# freqrank is the token's line number in the frequency-sorted .txt file, so 0
# is the most common token in the corpus. Lower rank means more common, which
# in practice also means denser neighbourhood and easier search -- which is why
# you want it as a control variable when correlating topic against recall.
# ----------------------------------------------------------------------------

log("writing word and frequency-rank maps...")


def dump_maps(prefix, txt_lines):
    """txt_lines: .txt line numbers, in the same order as the vectors."""
    with open(f"{OUT}/{prefix}.words.json", "w") as f:
        json.dump([words[int(t)] for t in txt_lines], f, ensure_ascii=False)
    with open(f"{OUT}/{prefix}.freqrank.json", "w") as f:
        json.dump([int(t) for t in txt_lines], f)


dump_maps("query.10K", test_idx)
dump_maps("query.english.1K", train_idx[english_rows])
dump_maps("learn.100K", train_idx[learn_rows])
dump_maps("validation.10K", train_idx[validation_rows])
dump_maps("base", train_idx[base_rows])

log("  done")


# ----------------------------------------------------------------------------
# Groundtruth
#
# Exact L2 over the base set. Note the metric: GloVe is conventionally an
# *angular* benchmark, but DARTH indexes and searches with L2, and these
# vectors are not unit-normalised. Using L2 here keeps groundtruth and search
# consistent, so recall is valid -- but the numbers are not comparable to
# published angular-GloVe results. State this explicitly in any writeup.
# ----------------------------------------------------------------------------

log(f"building exact index over {BASE_N} base vectors...")
index = faiss.IndexFlatL2(D)
index.add(base_vecs)
log("  done")


def groundtruth(name, vecs, ivecs_path, fvecs_path):
    log(f"groundtruth: {name} ({len(vecs)} queries, k={K})...")
    t0 = time.time()
    distances, indices = index.search(vecs, K)
    assert indices.shape == (len(vecs), K)
    assert distances.shape == (len(vecs), K)
    write_ivecs(ivecs_path, indices)
    write_fvecs(fvecs_path, distances)
    log(f"  done in {time.time() - t0:.1f}s")


# Ordered smallest-first so a mistake surfaces in minutes, not an hour.
groundtruth(
    "query_english", english_vecs,
    f"{OUT}/query.english.1K.groundtruth.k1000.ivecs",
    f"{OUT}/query.english.1K.groundtruth.k1000.fvecs",
)
groundtruth(
    "validation", validation_vecs,
    f"{OUT}/validation.groundtruth.10K.k1000.ivecs",
    f"{OUT}/validation.groundtruth.10K.k1000.fvecs",
)
groundtruth(
    "query", query_vecs,
    f"{OUT}/query.groundtruth.10K.k1000.ivecs",
    f"{OUT}/query.groundtruth.10K.k1000.fvecs",
)
groundtruth(
    "learn", learn_vecs,
    f"{OUT}/learn.groundtruth.100K.k1000.ivecs",
    f"{OUT}/learn.groundtruth.100K.k1000.fvecs",
)


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

manifest = {
    "source": "ann-benchmarks glove-100-angular.hdf5 (GloVe Twitter 27B, 100d)",
    "words_from": "glove.twitter.27B.100d.txt, exact vector match, 100% hit rate",
    "seed": SEED,
    "dim": D,
    "k": K,
    "metric": "L2, not angular -- see note in script",
    "splits": {
        "base": int(BASE_N),
        "learn": int(len(learn_rows)),
        "validation": int(len(validation_rows)),
        "query": int(len(query_vecs)),
        "query_english": int(len(english_rows)),
    },
    "base_filename": BASE_FILENAME,
    "english_query_freq_head": FREQ_HEAD,
    "notes": [
        "learn, validation and query_english are held out of base.",
        "query is the untouched hdf5 test split.",
        "freqrank = line number in the frequency-sorted .txt file, 0 = most common.",
        "frequency is confounded with neighbourhood density; control for it.",
    ],
}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

log("manifest written")
log("")
log("=" * 68)
log("done")
log("=" * 68)
log("")
log("Next: point VectorDataLoader.cpp at the base file, then rebuild.")
log("")
log("  cd /proj/batchvectordb-PG0/DARTH/DARTH_testing")
log(f"  sed -i 's|GLOVE100/base\\.[0-9]*\\.fvecs|GLOVE100/{BASE_FILENAME}|' \\")
log("      hnsw-test/VectorDataLoader.cpp")
log("  grep -n 'GLOVE100/base' hnsw-test/VectorDataLoader.cpp")
log("  make -C build -j$(nproc) faiss")
log("  make -C build -j$(nproc) ivf_test hnsw_test")
log("")
