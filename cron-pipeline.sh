#!/bin/bash
# ai-writings Vectorize Pipeline — Cron Script
# =============================================
# Runs the full update + sync pipeline.
# Designed to be run hourly via cron.
#
# Cron installation (Casey must approve):
#   0 * * * * /home/eileen/projects/ai-writings-vectorizer/cron-pipeline.sh >> /home/eileen/projects/ai-writings-vectorizer/cron.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"
echo "ai-writings Vectorize Pipeline"
echo "=========================================="

# Step 1: Update local embeddings (only new/modified files)
echo ""
echo "[1/2] Updating local embeddings..."
python3 vectorize.py --update

# Step 2: Sync to Cloudflare Vectorize (incremental)
echo ""
echo "[2/2] Syncing to Cloudflare Vectorize..."
python3 sync_to_cloudflare.py --update

echo ""
echo "✅ Pipeline complete."
