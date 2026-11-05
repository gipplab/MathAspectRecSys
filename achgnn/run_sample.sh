#!/usr/bin/env bash
# End-to-end smoke test on a small slice: 150 SilverRiM seeds + all GoldRiM pairs, their documents and up to 3000
# cited documents, TF-IDF features, 2 short epochs on CPU. Takes ~10-15 minutes, mostly downloading.
#
#   bash achgnn/run_sample.sh [path/to/mathRec_dataset.csv] [encoder: tfidf|hf] [device: auto|cpu|cuda]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV=${1:-$ROOT/achgnn/data/mathRec_dataset.csv}
ENCODER=${2:-tfidf}
DEVICE=${3:-auto}
WORK=$ROOT/achgnn/work_sample
ZB=$ROOT/achgnn/zbmath_sample

python "$ROOT/achgnn/dataset.py" --csv "$CSV" --work "$WORK" --sample-seeds 150
(cd "$ROOT/zbmath_hetnet" && python fetch_ids.py --ids-file "$WORK/dataset_ids.txt" --out-dir "$ZB" \
    --citation-hops 1 --max-per-hop 3000)
(cd "$ROOT/zbmath_hetnet" && python build_graph.py --out-dir "$ZB" --with-text)
python "$ROOT/achgnn/prepare_graph.py" --graph-dir "$ZB/graph" --work "$WORK"
if [ "$ENCODER" = hf ]; then
    python "$ROOT/achgnn/embed.py" --work "$WORK" --encoder hf --model Qwen/Qwen3-Embedding-0.6B --device "$DEVICE"
else
    python "$ROOT/achgnn/embed.py" --work "$WORK" --encoder tfidf --dim 256
fi
python "$ROOT/achgnn/train.py" --work "$WORK" --run sample --epochs 2 --batch-size 64 --device "$DEVICE"
SEED=$(python -c "import sys, pandas as pd; p = pd.read_csv(sys.argv[1], dtype=str); print(p[p.gold == 'True'].seed.iloc[0])" \
    "$WORK/pairs.csv")
python "$ROOT/achgnn/recommend.py" --work "$WORK" --run sample --zbl "$SEED" --top 5 --device "$DEVICE"
