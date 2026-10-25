#!/usr/bin/env bash
# Full pipeline: crawl all zbMATH Open documents and authors, then build the heterogeneous network.
# Re-running resumes the crawl where it stopped.
set -e
cd "$(dirname "$0")"
OUT=${1:-output}

python crawl.py all --out-dir "$OUT" --workers 4
python build_graph.py --out-dir "$OUT"
python export_graph.py --out-dir "$OUT" --format pyg || echo "PyG export skipped (needs torch + torch_geometric)"
