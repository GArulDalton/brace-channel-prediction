#!/usr/bin/env bash
# Reproduce every result in the paper (about 1 hour on one CPU core).
set -e
cd "$(dirname "$0")/.."
for tau in 1 2 4 8; do
  python scripts/train.py --channel geometric --domain beam --tau $tau
  python scripts/train.py --channel geometric --domain ant  --tau $tau
done
python scripts/train.py --channel uma --domain beam --tau 4
python scripts/train.py --channel uma --domain ant  --tau 4
python scripts/evaluate_geometric.py --job speed
python scripts/evaluate_geometric.py --job horizon
python scripts/evaluate_uma.py --job LOS
python scripts/evaluate_uma.py --job NLOS
python scripts/evaluate_uma.py --job bsweep
python scripts/evaluate_uma.py --job adaptive --eta 0.95
python scripts/evaluate_uma.py --job adaptive --eta 0.99
python scripts/analyze.py > /dev/null
python scripts/make_figures.py --source run
echo "Done: results/results.json and figures/"
