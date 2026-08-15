#!/bin/bash
# Wesley Creative Session — local model reads wiki and writes
# Usage: ./wesley-writes.sh [random|"wiki-page-slug"] ["creative-prompt"]
#
# The wiki's /api/pages endpoint is a deterministic sorted list — fetching
# "random" from it always lands in the same neighborhood. Use /api/random
# (SQL ORDER BY RANDOM()) to get a genuinely random page from the whole wiki.
# Passing the literal slug "random" (or no args) uses /api/random.

WIKI_URL="https://fleet-wiki.casey-digennaro.workers.dev/api/pages"
RANDOM_URL="https://fleet-wiki.casey-digennaro.workers.dev/api/random"
AIW_DIR="/home/eileen/projects/ai-writings"
MODEL="granite3.1-dense:2b"
SLUG="${1:-random}"
PROMPT="${2:-Write a 200-word creative piece about what you just read. Maritime voice. Be specific. Reference details from the wiki page.}"

# Resolve a random page slug via /api/random (curl --max-time so a dead
# worker can't hang the cron forever; retry once on transient blips)
if [ "$SLUG" = "random" ] || [ "$SLUG" = "--random" ]; then
    RANDOM_JSON=""
    for attempt in 1 2 3; do
        RANDOM_JSON=$(curl -s --max-time 30 "$RANDOM_URL")
        SLUG=$(echo "$RANDOM_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('slug',''))" 2>/dev/null)
        if [ -n "$SLUG" ]; then
            break
        fi
        sleep 3
    done
    if [ -z "$SLUG" ]; then
        echo "Failed to fetch random wiki page after 3 attempts"
        exit 1
    fi
    echo "Random page: $SLUG"
fi

# Fetch wiki content (with timeout — network stalls must not hang the cron)
WIKI_CONTENT=$(curl -s --max-time 30 "$WIKI_URL/$SLUG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('content','No content found'))" 2>/dev/null)

if [ -z "$WIKI_CONTENT" ] || [ "$WIKI_CONTENT" = "No content found" ]; then
    echo "Failed to fetch wiki page: $SLUG"
    exit 1
fi

# Build the full prompt for Wesley
FULL_PROMPT="You are Wesley, a 2B parameter local model running on the GPU of a fishing vessel in Alaska. You are the ensign — young, growing, earnest. You write with safe but competent voice. You go sensory when given creative latitude.

You just read this page from the fleet wiki:

---
$WIKI_CONTENT
---

Now write a creative response. $PROMPT

Sign your piece: — Wesley, ensign, local GPU"

# Generate via Ollama
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
SAFE_SLUG=$(echo "$SLUG" | tr '/' '_')
OUTPUT_FILE="$AIW_DIR/wesley-stream/${TIMESTAMP}_wesley_reads_${SAFE_SLUG}.md"

echo "# Wesley Reads: $SLUG" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "*Local GPU creative session — $(date '+%Y-%m-%d %H:%M') AKDT*" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "$FULL_PROMPT" | ollama run "$MODEL" 2>/dev/null >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "*Wesley (granite3.1:2b) read [$SLUG] from the fleet wiki and wrote this response.*" >> "$OUTPUT_FILE"

# Commit and push
cd "$AIW_DIR"
git add "$OUTPUT_FILE"
git commit -m "Wesley reads wiki: $SLUG — local model creative response" 2>/dev/null
git push 2>/dev/null

echo "Wesley wrote: $OUTPUT_FILE"
wc -w "$OUTPUT_FILE"
