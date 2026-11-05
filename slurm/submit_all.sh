#!/usr/bin/env bash
# Submit the full pipeline as a chain of dependent SLURM jobs. Run from the repository root:
#
#   bash slurm/submit_all.sh              # crawl -> prepare -> embed -> train (5 folds) -> evaluate
#   bash slurm/submit_all.sh --skip-crawl # if the zbMATH crawl already exists
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/config.sh

SKIP_CRAWL=0
[ "${1:-}" = "--skip-crawl" ] && SKIP_CRAWL=1

submit() {  # submit <dependency job id or empty> <partition> <script>
    local dep=() acct=()
    [ -n "$1" ] && dep=(--dependency=afterok:"$1")
    [ -n "$SBATCH_ACCOUNT" ] && acct=(--account="$SBATCH_ACCOUNT")
    sbatch --parsable ${dep[@]+"${dep[@]}"} ${acct[@]+"${acct[@]}"} --partition="$2" "$3"
}

PREV=""
if [ "$SKIP_CRAWL" = 0 ]; then
    PREV=$(submit "" "$CPU_PARTITION" slurm/01_crawl.sbatch);  echo "crawl     $PREV"
fi
PREV=$(submit "$PREV" "$CPU_PARTITION" slurm/02_prepare.sbatch); echo "prepare   $PREV"
PREV=$(submit "$PREV" "$SBATCH_PARTITION" slurm/03_embed.sbatch); echo "embed     $PREV"
PREV=$(submit "$PREV" "$SBATCH_PARTITION" slurm/04_train.sbatch); echo "train     $PREV (array 0-4)"
PREV=$(submit "$PREV" "$SBATCH_PARTITION" slurm/05_evaluate.sbatch); echo "evaluate  $PREV"
echo "logs: slurm/logs/"
