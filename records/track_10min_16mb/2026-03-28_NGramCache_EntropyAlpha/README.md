# N-gram Cache + Entropy-Adaptive Alpha

Based on SOTA `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` (1.1194 BPB).

## Motivation

Inspired by the Engram paper (arxiv 2601.07372v1): static memorized knowledge should live in O(1) lookup tables, not neural weights. At 16MB scale the analogue is a backward-looking n-gram count cache at eval time. The competition's #933 achieves 0.0804 BPB this way. Entropy-adaptive α weighting (from issue #140) is the routing mechanism: uncertain tokens rely on the neural model, low-entropy tokens defer to the n-gram table.

**TurboQuant (arxiv 2504.19874) is not included here.** TurboQuant compresses a persistent KV cache across autoregressive decode steps. This eval does prefill-only forward passes: each sliding window processes a fresh 2048-token sequence in one shot via Flash Attention 3, which never materializes a persistent KV cache. There is nothing for TurboQuant to compress. It would only become relevant if the eval used autoregressive generation with a growing KV buffer.

## Changes vs SOTA

All changes are **eval-only** (score-first, legal). Training is identical.

- `NGramCache` class: per-order count tables (min_order..max_order), polynomial rolling hash
- Injected into `eval_val_sliding_ttt`: query before scoring, update after (score-first preserved)
- Linear probability mixing (default) or log-linear mixing (`NGRAM_LOGISTIC_MIX=1`)
- Entropy-adaptive α (default) or fixed α

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NGRAM_ENABLED` | `0` | Master switch — set to `1` to activate |
| `NGRAM_MAX_ORDER` | `7` | Maximum n-gram order |
| `NGRAM_MIN_ORDER` | `2` | Minimum order for backoff |
| `NGRAM_ALPHA_MODE` | `entropy` | `entropy` (adaptive) or `fixed` |
| `NGRAM_ALPHA_FIXED` | `0.5` | Used when `NGRAM_ALPHA_MODE=fixed` |
| `NGRAM_ALPHA_MIN` | `0.05` | Min α for entropy mode |
| `NGRAM_ALPHA_MAX` | `0.60` | Max α for entropy mode |
| `NGRAM_ENTROPY_BASE` | `4.0` | H₀ in α = α_min + (α_max−α_min)·σ(2(H−H₀)) |
| `NGRAM_SMOOTHING` | `0.01` | Laplace smoothing denominator |
| `NGRAM_LOGISTIC_MIX` | `0` | `1` = log-linear blend + renorm; `0` = linear prob blend |
| `NGRAM_COMP_TRAIN` | `0` | Complementary training (not yet implemented) |
| `NGRAM_COMP_LAMBDA` | `0.3` | Down-weighting strength for comp training |

## Test Matrix

Run SOTA command + add the n-gram flags below. All other hyperparameters unchanged.

### Baseline (confirm SOTA reproduces)
```bash
# No NGRAM_ENABLED — identical to SOTA
```

### Test A: 5-gram, entropy adaptive, linear mix (expected ~−0.07 BPB)
```bash
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=5 NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA_MODE=entropy NGRAM_ALPHA_MIN=0.05 NGRAM_ALPHA_MAX=0.60 NGRAM_ENTROPY_BASE=4.0
```

### Test B: 7-gram, entropy adaptive (expected small additional gain over A)
```bash
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA_MODE=entropy NGRAM_ALPHA_MIN=0.05 NGRAM_ALPHA_MAX=0.60 NGRAM_ENTROPY_BASE=4.0
```

### Test C: 7-gram, fixed α=0.5 (ablation vs entropy adaptive)
```bash
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA_MODE=fixed NGRAM_ALPHA_FIXED=0.5
```

### Test D: 7-gram, log-linear (logistic-domain) mix (ablation vs linear)
```bash
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA_MODE=entropy NGRAM_LOGISTIC_MIX=1
```

### Test E: 5-gram only (2-gram through 5-gram, no 6-7)
```bash
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=5 NGRAM_MIN_ORDER=5 \
NGRAM_ALPHA_MODE=entropy
```

### Test F: high α ceiling (try to squeeze more from n-gram)
```bash
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA_MODE=entropy NGRAM_ALPHA_MIN=0.05 NGRAM_ALPHA_MAX=0.80 NGRAM_ENTROPY_BASE=4.0
```

## Full Run Command (Test A)

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 \
TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 VAL_LOSS_EVERY=500 \
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=5 NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA_MODE=entropy NGRAM_ALPHA_MIN=0.05 NGRAM_ALPHA_MAX=0.60 NGRAM_ENTROPY_BASE=4.0 \
SEED=1337 torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-28_NGramCache_EntropyAlpha/train_gpt.py
# repeat for SEED=42, SEED=2025
```

## Legality Notes

Per @NoesisGenesis / @valerio-oai's formal criteria (issue #677):
- **Full distribution**: counts are stored for all 1024 vocab tokens + total; Laplace smoothing ensures every token has a non-zero probability → `sum P(token | ctx) = 1` ✓
- **Score-first**: `query_positions` is called before `update_window` for every window ✓
- **No target leak**: entropy `H` is computed from the neural log-probs only; `tgt_np` is only accessed AFTER the blended distribution is fully determined ✓
- **Single pass**: sequential chunk/window processing, no rescoring, no multi-pass selection ✓
- **Hashing is storage only**: the `_hash_ctx` polynomial hash is an index into a dict; it does not affect the distribution (the full count array is always over all vocab tokens) ✓

The illegal submissions on #677 failed because they: (a) only tracked counts for observed tokens and computed NLL directly without ever defining a full normalized distribution, or (b) used the target token to pick between model and n-gram (teacher forcing).

## Notes

- N-gram cache is per-rank (no cross-rank sync). Each rank builds statistics from the windows it scores. This is a valid approximation; each rank processes sequential tokens within each chunk.
- Memory: 7-gram cache for ~5M tokens ≈ a few hundred MB of Python dicts. Monitor GPU/CPU memory.
- The cache is not persisted to disk; it rebuilds each eval run (by design for score-first legality).
- TurboQuant KV compression (for faster TTT with same memory) is a future addition if KV memory proves to be a bottleneck.
