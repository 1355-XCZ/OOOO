#!/bin/bash
# ==============================================================
# Submit RQ2 Combined: SER-F1 + Entropy evaluation -> Plot
#
# Phase 1: Parallel eval jobs (SER + entropy, 3 SSL models each)
# Phase 2: Plot job (depends on all eval jobs)
#
# Usage:
#   bash paper_pipeline/slurm/submit_rq2_combined.sh [--dry-run]
# ==============================================================

set -e
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_SH="${SLURM_DIR}/env.sh"
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
    esac
done

submit() {
    local cmd="$1"
    if [ -n "$DRY_RUN" ]; then
        echo "  [DRY RUN] $cmd" >&2
        echo "DRYRUN_$$"
    else
        echo "  Submitting: $cmd" >&2
        local output
        output=$(eval "$cmd")
        local job_id=$(echo "$output" | grep -oP '\d+' | head -1)
        echo "    -> Job ID: $job_id" >&2
        echo "$job_id"
    fi
}

echo "============================================"
echo "  RQ2 Combined: SER-F1 + Entropy"
echo "============================================"

echo ""
echo "--- Phase 1: SER-F1 Evaluation (36 GPU tasks) ---"
SER_JID=$(submit "sbatch --array=0-35 --parsable ${SLURM_DIR}/rq2_ser_eval.slurm")

echo ""
echo "--- Phase 1: Entropy Evaluation (36 GPU tasks) ---"
ENT_JID=$(submit "sbatch --array=0-35 --parsable ${SLURM_DIR}/rq2_entropy_eval.slurm")

echo ""
echo "--- Phase 2: Plot (CPU, after all eval jobs) ---"
PLOT_JID=$(submit "sbatch --dependency=afterok:${SER_JID}:${ENT_JID} --parsable \
    --job-name=rq2_plot \
    --partition=sapphire \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --output=paper_pipeline/slurm/logs/rq2_plot_%j.out \
    --error=paper_pipeline/slurm/logs/rq2_plot_%j.err \
    --wrap='source ${ENV_SH} && python -m paper_pipeline.figures.rq2_combined'")

echo ""
echo "============================================"
echo "  Jobs submitted!"
echo "  SER Eval:     ${SER_JID} (36 tasks)"
echo "  Entropy Eval: ${ENT_JID} (36 tasks)"
echo "  Plot:         ${PLOT_JID}"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Output:  results/paper_figures_rq/rq2_combined_hubert.png"
echo "           results/paper_figures_rq/rq2_combined_wavlm.png"
echo "============================================"
