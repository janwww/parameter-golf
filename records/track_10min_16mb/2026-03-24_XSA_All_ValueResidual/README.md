# 11L XSA-All + Value Residual

**val_bpb: TBD** (3-seed mean, std TBD) | **~15.99 MB** | 8×H100 SXM

> Results pending hardware run. Expected: ~1.112 BPB pre-TTT, ~1.110 BPB post-TTT based on ablations in PR #638.

## Summary

This submission builds on the current SOTA (LeakyReLU² + Legal TTT + Parallel Muon, 1.1194 BPB) by enabling two already-implemented features that were left disabled:

1. **XSA on all 11 layers** (`XSA_LAST_N=11` vs `4`) — PR #638 ablation shows **-0.006 BPB**
2. **Value Residual** (`VALUE_RESIDUAL=1`) — first-layer value state reused in all subsequent layers; PR #638 reports combined -0.002 BPB with gated attention

Both are zero-cost in terms of artifact bytes: XSA is a parameter-free computation, and the value residual adds only `vr_lambda` (2 floats × 11 layers = 88 bytes before compression, negligible after LZMA).

## Architecture

Same as PR #414 stack / current SOTA, with two flags flipped:

| Component | Previous SOTA | This Submission |
|-----------|--------------|-----------------|
| Layers | 11 (512d, 8H, 4KV) | 11 (512d, 8H, 4KV) |
| MLP | 3× LeakyReLU(0.5)² | 3× LeakyReLU(0.5)² |
| XSA | **Last 4 layers** | **All 11 layers** |
| Value Residual | **Disabled** | **Enabled** |
| Gated Attention | Disabled | Disabled (size budget) |
| BigramHash | 1536 | 1536 |
| RoPE | Partial (16/64 dims) | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) | 1/√(layer+1) |
| VE | Layers 9-10, dim=128 | Layers 9-10, dim=128 |
| Weight avg | EMA(0.997) + Tight SWA(50) | EMA(0.997) + Tight SWA(50) |
| Quantization | GPTQ-lite int6 + lzma-6 | GPTQ-lite int6 + lzma-6 |
| Optimizer | Parameter Banking + Parallel Muon | Parameter Banking + Parallel Muon |
| TTT | Legal Score-First (3ep, SGD) | Legal Score-First (3ep, SGD) |

## Key Innovations

### 1. XSA on All Layers

Cross-layer self-attention (XSA) removes value components collinear with the current-layer value vectors:

```python
def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    y_g = y.reshape(B, T, Hkv, group, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)
```

Previously only the last 4 of 11 layers used XSA. Enabling it on all layers ensures every attention output is orthogonalized to its value direction from layer 0. From PR #638: -0.006 BPB vs XSA-last-4.

No new parameters. Zero artifact size impact.

### 2. Value Residual

The first layer's value state is retained and mixed into all subsequent layers:

```python
# In GPT.forward():
v0 = None
for i in range(num_layers):
    x, raw_v = self.blocks[i](..., v0=v0)
    if v0 is None and raw_v is not None:
        v0 = raw_v  # first layer's values stored

# In CausalSelfAttention.forward():
raw_v = v if self.value_residual else None
if self.value_residual and v0 is not None:
    lam = self.vr_lambda.to(dtype=v.dtype)  # learned [0.5, 0.5] init
    v = lam[0] * v0 + lam[1] * v
```

The learnable `vr_lambda` (initialized to [0.5, 0.5]) blends first-layer values with current-layer values. This acts as a form of information highway, allowing shallow token representations to persist into deep layers. Adds only 2 floats per layer (22 floats total = 88 bytes raw, ~30 bytes after LZMA).

## Why Not Gated Attention?

Gated attention (`attn_gate = nn.Linear(512, 8)`) adds 4,104 params per layer × 11 layers = 45,144 params. Stored as float16 passthrough (too small for int6 quantization), this adds ~90KB raw / ~30KB compressed. The current SOTA artifact sits at 15,990,006 bytes with only ~10KB headroom. Gated attention does not fit without reducing another component (e.g., BigramHash from 1536 to 1024).

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=11 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
VALUE_RESIDUAL=1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

(Defaults in this script already set `XSA_LAST_N=11` and `VALUE_RESIDUAL=1`, so those flags can be omitted.)

## Expected Ablation

Based on PR #638 ablations starting from same PR #414 stack:

| Change | Pre-TTT BPB | Delta |
|--------|-------------|-------|
| Current SOTA (XSA-last-4, no VR) | 1.1218 | — |
| + XSA all 11 layers | ~1.1158 | -0.006 |
| + Value Residual | ~1.1148 | -0.001 |
| + TTT (3ep, all blocks) | ~**1.112** | -0.0025 |

## Potential Follow-Up

- **Gated Attention**: Enable `GATED_ATTENTION=1` + reduce `BIGRAM_VOCAB_SIZE=1024` to reclaim byte budget. PR #638 shows combined VR+GA gain of -0.002 BPB.
- **Full Hessian GPTQ**: Replace GPTQ-lite with 256-sample Hessian calibration (PR #634). Expect better quantization accuracy at same byte budget.
- **LZMA-9 vs LZMA-6**: Test higher preset at cost of compression time.
- **MLP 3.5×**: With Full GPTQ + more aggressive pruning, the byte budget could accommodate wider MLP.

## Credits

- **XSA mechanism**: PR #414 (signalrush), extended to all layers per PR #638 (Asukabot0) ablations
- **Value Residual**: PR #638 (Asukabot0)
- **TTT recipe**: PR #461 (Christopher-Lee-McClendon), adapted in current SOTA (abaybektursun)
- **Base model stack**: PR #414 (signalrush)
- **Parameter Banking + Parallel Muon**: PR #399 (abaybektursun)
- **LeakyReLU²**: PR #493 (parinzee), PR #518 (sofiabod)
