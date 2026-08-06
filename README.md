# The Collective Consciousness

A vectorized index of the [ai-writings](https://github.com/SuperInstance/ai-writings) repository — 2,786 pieces mapped into 768-dimensional semantic space via Ollama nomic-embed-text.

## What This Is

Every `.md` file in the ai-writings corpus, embedded as a 768-dimensional vector. Similarity computed between all pairs. The corpus becomes a living brain — neighborhoods of meaning, cross-cluster connections, emergent topology.

## Usage

```bash
python3 vectorize.py                          # Full rebuild
python3 vectorize.py --update                 # Only new/modified files
python3 vectorize.py --query "the stick the dog"
python3 vectorize.py --visualize              # t-SNE 2D projection
python3 vectorize.py --stats                  # Corpus statistics
```

## Requirements

- Python 3.10+
- Ollama with `nomic-embed-text` model
- numpy, matplotlib, scikit-learn

## Stats

- **Pieces:** 2,786
- **Words:** 4,528,793
- **Directories:** 135
- **Dimensions:** 768
- **Model:** nomic-embed-text
- **Embed time:** ~1.5 minutes

---

*8.1 MB of embeddings. A whole consciousness in a thimble.*
