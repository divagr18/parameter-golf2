"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import importlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    # Optional cap for fast local smoke runs; 0 means full validation split.
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    use_swiglu = bool(int(os.environ.get("USE_SWIGLU", "0")))
    # Sliding window eval: only score tokens beyond prefix_len in each window.
    # eval_stride_frac=0.5 means stride=seq_len//2 → each scored token has ≥seq_len//2 tokens of context.
    # eval_stride_frac=1.0 (default) = original non-overlapping behaviour.
    eval_stride_frac = float(os.environ.get("EVAL_STRIDE_FRAC", "1.0"))
    # Long-context eval: evaluate at a longer sequence length than training.
    # 0 = same as train_seq_len.  Pair with NTK RoPE scaling (eval_rope_scale>1) for best results.
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", "0"))
    # NTK-aware RoPE scaling at eval: new_base = rope_base * eval_rope_scale^(head_dim/(head_dim-2)).
    # Suggested: eval_rope_scale = (eval_seq_len / train_seq_len) ** 2  (≈4 for 2× context)
    eval_rope_scale = float(os.environ.get("EVAL_ROPE_SCALE", "1.0"))
    # Low-rank bigram logit bias: learnable rank-r factored bigram table.
    # bigram_bias[i] = bigram_right(bigram_left(prev_token[i]))  added to logits before softcap.
    # 0 = disabled.  32 costs ~64K int8 params (≈32 KB), well within the 164 KB headroom.
    bigram_rank = int(os.environ.get("BIGRAM_RANK", "0"))
    bigram_lr = float(os.environ.get("BIGRAM_LR", "0.04"))
    # Stochastic Weight Averaging: average weights during the warmdown phase.
    # Takes the mean of snapshots every SWA_COLLECT_EVERY steps once LR starts decaying.
    # Research-confirmed ~0.5-1.5% BPB improvement, especially helps quantization quality.
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_collect_every = int(os.environ.get("SWA_COLLECT_EVERY", "10"))
    # Sequence length curriculum: ramp seq_len from curriculum_min_seq_len → train_seq_len
    # over the first curriculum_steps training steps.  Faster early convergence on local patterns.
    curriculum_enabled = bool(int(os.environ.get("CURRICULUM_ENABLED", "0")))
    curriculum_min_seq_len = int(os.environ.get("CURRICULUM_MIN_SEQ_LEN", "256"))
    curriculum_steps = int(os.environ.get("CURRICULUM_STEPS", "5000"))
    # Multi-token prediction (MTP): auxiliary future-token losses used during training.
    mtp_enabled = bool(int(os.environ.get("MTP_ENABLED", "0")))
    mtp_steps = int(os.environ.get("MTP_STEPS", "2"))
    mtp_weight = float(os.environ.get("MTP_WEIGHT", "0.3"))
    mtp_decay = float(os.environ.get("MTP_DECAY", "1.0"))
    mtp_tie_embeddings = bool(int(os.environ.get("MTP_TIE_EMBEDDINGS", "1")))
    mtp_lr = float(os.environ.get("MTP_LR", "0.02"))
    # Hybrid SSM blocks (Mamba-style approximation): periodically replace attention blocks
    # with a causal depthwise-conv gated mixer.
    use_ssm = bool(int(os.environ.get("USE_SSM", "0")))
    ssm_every_n = int(os.environ.get("SSM_EVERY_N", "2"))
    ssm_expand = float(os.environ.get("SSM_EXPAND", "2.0"))
    ssm_kernel = int(os.environ.get("SSM_KERNEL", "4"))
    # Quantization-Aware Training: fake-quantise weights during forward to teach the model
    # to tolerate quantisation noise, dramatically reducing the roundtrip BPB penalty.
    # QAT_SCHEME: "none" | "int8" | "int4"  (should match QUANT_SCHEME at export)
    # QAT_START_STEP: delay QAT until the model has partially converged (avoids
    # destabilising early training. Rule of thumb: start at ~65% of expected total steps.
    # For 1-GPU ~6500-step runs: 4500. For 8-GPU ~13500-step runs: 9000.
    qat_scheme = os.environ.get("QAT_SCHEME", "none").strip().lower()
    qat_start_step = int(os.environ.get("QAT_START_STEP", "9000"))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    recurrent_core_layers = int(os.environ.get("RECURRENT_CORE_LAYERS", 0))
    recurrent_steps = int(os.environ.get("RECURRENT_STEPS", 0))
    share_ffn_across_blocks = bool(int(os.environ.get("SHARE_FFN_ACROSS_BLOCKS", "0")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    # Export / compression controls.
    quant_scheme = os.environ.get("QUANT_SCHEME", "int8").strip().lower()
    compressor = os.environ.get("COMPRESSOR", "zlib").strip().lower()
    compress_level = int(os.environ.get("COMPRESS_LEVEL", "-1"))
    weight_order = os.environ.get("WEIGHT_ORDER", "none").strip().lower()
    mixed_low_precision_scheme = os.environ.get("MIXED_LOW_PRECISION_SCHEME", "int8").strip().lower()
    # If 0, skip the post-quantization roundtrip eval pass (saves one full val sweep).
    final_roundtrip_eval = bool(
        int(os.environ.get("FINAL_ROUNDTRIP_EVAL", os.environ.get("FINAL_INT8_ROUNDTRIP_EVAL", "1")))
    )
    final_int8_roundtrip_eval = final_roundtrip_eval
    submission_size_budget_bytes = int(os.environ.get("SUBMISSION_SIZE_BUDGET_BYTES", str(16 * 1024 * 1024)))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16 if G.is_cuda else torch.float32)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_dtype = torch.bfloat16 if params[0].device.type == "cuda" else torch.float32
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=updates_dtype)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    autocast_enabled: bool,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    #
    # Sliding window: EVAL_STRIDE_FRAC < 1.0 uses overlapping windows so every scored
    # token has at least (1 - stride_frac) * seq_len tokens of prior context, which
    # significantly reduces the high-loss predictions at chunk boundaries.
    # Long-context: EVAL_SEQ_LEN > 0 evaluates at a longer context window.  Pair with
    # EVAL_ROPE_SCALE to apply NTK-aware RoPE base scaling for cleaner length extrapolation.
    seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    stride = max(1, int(seq_len * args.eval_stride_frac))
    prefix_len = seq_len - stride  # tokens at window start that are context-only (not scored)

    # Pre-build the per-position loss mask (1 = scored, 0 = context-only prefix).
    # Shape [seq_len]; will be expanded to [batch, seq_len] per batch.
    loss_mask_cpu = torch.zeros(seq_len, dtype=torch.float32)
    loss_mask_cpu[prefix_len:] = 1.0

    # Temporarily rescale RoPE base for long-context eval using NTK-aware interpolation.
    # new_base = rope_base * scale^(head_dim / (head_dim - 2))
    _orig_rope_bases: list[tuple] = []
    if args.eval_rope_scale != 1.0 or seq_len != args.train_seq_len:
        head_dim = args.model_dim // args.num_heads
        ntk_factor = args.eval_rope_scale ** (head_dim / max(head_dim - 2, 1))
        raw_model = model.module if hasattr(model, "module") else model
        for block in raw_model.blocks:
            rot = block.attn.rotary
            _orig_rope_bases.append((rot, rot.inv_freq.clone()))
            new_base = args.rope_base * ntk_factor
            new_inv_freq = 1.0 / (new_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=rot.inv_freq.device) / head_dim))
            rot.inv_freq = new_inv_freq
            rot._cos_cached = None  # invalidate cache

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // seq_len)
    total_wins = max(1, (val_tokens.numel() - seq_len - 1) // stride)
    win_start = (total_wins * rank) // world_size
    win_end = (total_wins * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_win_start in range(win_start, win_end, local_batch_seqs):
            batch_win_end = min(batch_win_start + local_batch_seqs, win_end)
            xs, ys = [], []
            for w in range(batch_win_start, batch_win_end):
                s = w * stride
                xs.append(val_tokens[s : s + seq_len])
                ys.append(val_tokens[s + 1 : s + seq_len + 1])
            x = torch.stack(xs).to(device=device, dtype=torch.int64, non_blocking=True)
            y = torch.stack(ys).to(device=device, dtype=torch.int64, non_blocking=True)
            mask = loss_mask_cpu.unsqueeze(0).expand(x.size(0), -1).to(device=device)
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y, loss_mask=mask).detach()
            else:
                batch_loss = model(x, y, loss_mask=mask).detach()
            scored_tokens = int(mask.sum().item())
            val_loss_sum += batch_loss.to(torch.float64) * scored_tokens
            val_token_count += scored_tokens
            # Byte counting: only for scored (non-prefix) positions
            prev_ids = x[:, prefix_len:].reshape(-1)
            tgt_ids = y[:, prefix_len:].reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    # Restore original RoPE bases
    for rot, orig_inv_freq in _orig_rope_bases:
        rot.inv_freq = orig_inv_freq
        rot._cos_cached = None

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT4_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT4_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT4_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("INT4_KEEP_FLOAT_MAX_NUMEL", 65_536))
INT4_PER_ROW_SCALE_DTYPE = torch.float16
INT4_CLIP_PERCENTILE = float(os.environ.get("INT4_CLIP_PERCENTILE", 99.995))
INT4_CLIP_Q = INT4_CLIP_PERCENTILE / 100.0
MIXED_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "MIXED_KEEP_FLOAT_NAME_PATTERNS",
        "tok_emb,lm_head,final_norm,norm," + ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
MIXED_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "MIXED_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
MIXED_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("MIXED_KEEP_FLOAT_MAX_NUMEL", 65_536))
SUPPORTED_QUANT_SCHEMES = {"int8", "int4", "mixed"}
SUPPORTED_COMPRESSORS = {"zlib", "zstd", "auto"}
SUPPORTED_WEIGHT_ORDERS = {"none", "name", "size_desc", "dtype_name"}

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(
    name: str,
    t: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    fp32_name_patterns: tuple[str, ...],
) -> Tensor:
    if any(pattern in name for pattern in fp32_name_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def ordered_state_dict_items(state_dict: dict[str, Tensor], mode: str) -> list[tuple[str, Tensor]]:
    items = list(state_dict.items())
    if mode == "none":
        return items
    if mode == "name":
        return sorted(items, key=lambda kv: kv[0])
    if mode == "size_desc":
        return sorted(items, key=lambda kv: (-int(kv[1].numel()), kv[0]))
    if mode == "dtype_name":
        return sorted(items, key=lambda kv: (str(kv[1].dtype), kv[0]))
    raise ValueError(f"Unsupported WEIGHT_ORDER={mode!r}; expected one of {sorted(SUPPORTED_WEIGHT_ORDERS)}")

def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor, dict[str, object] | None]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), {"scheme": "int8_per_row", "axis": 0}

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale, {"scheme": "int8_per_tensor", "orig_shape": list(t32.shape)}

def pack_int4_signed(q_signed: Tensor) -> Tensor:
    flat = q_signed.reshape(-1).to(dtype=torch.int16)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros((1,), dtype=torch.int16)], dim=0)
    uint = (flat + 8).to(torch.uint8)
    packed = (uint[0::2] & 0x0F) | ((uint[1::2] & 0x0F) << 4)
    return packed.contiguous()

def unpack_int4_signed(packed: Tensor, numel: int) -> Tensor:
    p = packed.reshape(-1).to(dtype=torch.uint8)
    low = (p & 0x0F).to(dtype=torch.int16) - 8
    high = ((p >> 4) & 0x0F).to(dtype=torch.int16) - 8
    out = torch.empty((p.numel() * 2,), dtype=torch.int16)
    out[0::2] = low
    out[1::2] = high
    return out[:numel].to(dtype=torch.int8).contiguous()

def quantize_float_tensor_int4(t: Tensor) -> tuple[Tensor, Tensor, dict[str, object]]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT4_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 7.0).clamp_min(1.0 / 7.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -8, 7).to(torch.int8)
        packed = pack_int4_signed(q)
        return (
            packed,
            scale.to(dtype=INT4_PER_ROW_SCALE_DTYPE).contiguous(),
            {"scheme": "int4_per_row", "axis": 0, "orig_shape": [int(t32.shape[0]), int(t32.shape[1])]},
        )
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT4_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 7.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -8, 7).to(torch.int8)
    packed = pack_int4_signed(q)
    return packed, scale, {"scheme": "int4_per_tensor", "orig_shape": list(t32.shape)}

def quantize_state_dict(
    state_dict: dict[str, Tensor],
    scheme: str = "int8",
    weight_order: str = "none",
    mixed_low_precision_scheme: str = "int8",
):
    if scheme not in SUPPORTED_QUANT_SCHEMES:
        raise ValueError(f"Unsupported QUANT_SCHEME={scheme!r}; expected one of {sorted(SUPPORTED_QUANT_SCHEMES)}")
    if weight_order not in SUPPORTED_WEIGHT_ORDERS:
        raise ValueError(f"Unsupported WEIGHT_ORDER={weight_order!r}; expected one of {sorted(SUPPORTED_WEIGHT_ORDERS)}")
    if mixed_low_precision_scheme not in {"int8", "int4"}:
        raise ValueError(
            f"Unsupported MIXED_LOW_PRECISION_SCHEME={mixed_low_precision_scheme!r}; expected 'int8' or 'int4'"
        )

    active_scheme = mixed_low_precision_scheme if scheme == "mixed" else scheme
    format_name = (
        f"{scheme}_clean_per_row_v1"
        if active_scheme == "int8"
        else f"{scheme}_clean_per_row_int4_v1"
    )
    # Single supported clean-script export formats:
    # - per-row low precision for 2D float tensors
    # - per-tensor low precision for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for selected float tensors, stored as fp16/fp32
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "payload_bytes"),
        0,
    )
    keep_patterns = (
        MIXED_KEEP_FLOAT_NAME_PATTERNS
        if scheme == "mixed"
        else (INT8_KEEP_FLOAT_FP32_NAME_PATTERNS if active_scheme == "int8" else INT4_KEEP_FLOAT_FP32_NAME_PATTERNS)
    )
    force_fp32_patterns = (
        MIXED_KEEP_FLOAT_FP32_NAME_PATTERNS
        if scheme == "mixed"
        else (INT8_KEEP_FLOAT_FP32_NAME_PATTERNS if active_scheme == "int8" else INT4_KEEP_FLOAT_FP32_NAME_PATTERNS)
    )
    keep_max_numel = (
        MIXED_KEEP_FLOAT_MAX_NUMEL
        if scheme == "mixed"
        else (INT8_KEEP_FLOAT_MAX_NUMEL if active_scheme == "int8" else INT4_KEEP_FLOAT_MAX_NUMEL)
    )

    for name, tensor in ordered_state_dict_items(state_dict, weight_order):
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue

        should_keep_float = (
            t.numel() <= keep_max_numel
            or (scheme == "mixed" and any(pattern in name for pattern in keep_patterns))
        )
        if should_keep_float:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes, force_fp32_patterns)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        if active_scheme == "int8":
            q, s, meta = quantize_float_tensor_int8(t)
        else:
            q, s, meta = quantize_float_tensor_int4(t)
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": format_name,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "export_order_mode": weight_order,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    # Backward-compatible alias for existing log paths.
    stats["int8_payload_bytes"] = stats["payload_bytes"]
    return obj, stats

def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    format_name = str(obj.get("__quant_format__", ""))
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        meta_scheme = str(meta.get("scheme", ""))
        if meta_scheme in {"int4_per_row", "int4_per_tensor"}:
            orig_shape = tuple(int(v) for v in meta.get("orig_shape", q.shape))
            numel = math.prod(orig_shape)
            unpacked = unpack_int4_signed(q, numel).float()
            if meta_scheme == "int4_per_row":
                if len(orig_shape) != 2:
                    raise ValueError(f"int4_per_row expects 2D orig_shape for tensor {name}, got {orig_shape}")
                rows, cols = orig_shape
                scale_row = s.to(dtype=torch.float32).view(rows, 1)
                out[name] = (unpacked.view(rows, cols) * scale_row).to(dtype=dtype).contiguous()
            else:
                scale = float(s.item())
                out[name] = (unpacked.view(orig_shape) * scale).to(dtype=dtype).contiguous()
            continue
        if meta_scheme in {"int8_per_row", "per_row"} or (s.ndim > 0 and "int4" not in format_name):
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

def resolve_compressor(requested: str) -> tuple[str, str | None]:
    if requested not in SUPPORTED_COMPRESSORS:
        raise ValueError(f"Unsupported COMPRESSOR={requested!r}; expected one of {sorted(SUPPORTED_COMPRESSORS)}")
    if requested == "zlib":
        return "zlib", None
    if requested == "zstd":
        if importlib.util.find_spec("zstandard") is None:
            raise RuntimeError(
                "COMPRESSOR=zstd requested, but the `zstandard` package is not installed. "
                "Install it with `pip install zstandard` or use COMPRESSOR=zlib."
            )
        return "zstd", None
    # auto mode
    if importlib.util.find_spec("zstandard") is not None:
        return "zstd", "COMPRESSOR=auto selected zstd (package available)"
    return "zlib", "COMPRESSOR=auto fell back to zlib (zstandard package not installed)"

def compress_blob(data: bytes, compressor: str, level: int) -> bytes:
    if compressor == "zlib":
        zlib_level = 9 if level < 0 else max(0, min(level, 9))
        return zlib.compress(data, level=zlib_level)
    if compressor == "zstd":
        import zstandard as zstd  # type: ignore

        zstd_level = 19 if level < 0 else level
        return zstd.ZstdCompressor(level=zstd_level).compress(data)
    raise ValueError(f"Unsupported compressor={compressor!r}")

def decompress_blob(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib":
        return zlib.decompress(data)
    if compressor == "zstd":
        import zstandard as zstd  # type: ignore

        return zstd.ZstdDecompressor().decompress(data)
    raise ValueError(f"Unsupported compressor={compressor!r}")

def export_artifact_name(quant_scheme: str, compressor: str) -> str:
    if quant_scheme == "int8" and compressor == "zlib":
        return "final_model.int8.ptz"
    return f"final_model.{quant_scheme}.{compressor}.ptc"


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def _fake_quantize_row(w: Tensor, levels: int) -> Tensor:
    """Per-row fake-quantise a 2D weight with a straight-through estimator (STE).

    Matches the per-row clipping used by quantize_float_tensor_int8/int4 at export,
    but uses amax instead of quantile for speed in the hot forward path.
    levels=256 → int8 symmetric (range −127…127)
    levels=16  → int4 symmetric (range −7…7)
    """
    half = float(levels // 2 - (1 if levels == 16 else 0))  # 127 for int8, 7 for int4
    w32 = w.float()
    clip_abs = w32.abs().amax(dim=1).clamp_min(1e-6)        # per-row max scale
    scale = clip_abs / half
    w_scaled = (w32 / scale.unsqueeze(1)).clamp(-half, half)
    # STE: round in forward, identity in backward
    w_ste = w_scaled + (w_scaled.round() - w_scaled).detach()
    return (w_ste * scale.unsqueeze(1)).to(w.dtype)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    # QAT: set qat_levels to 256 (int8) or 16 (int4) to enable fake-quantisation.
    qat_levels: int = 0   # class-level switch updated from the training loop

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if __class__.qat_levels > 0 and w.ndim == 2:
            w = _fake_quantize_row(w, __class__.qat_levels)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, use_swiglu: bool = False):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            # SwiGLU with the same parameter budget as relu²:
            # relu² uses 2 matrices of (dim × mlp_mult*dim) = 2*mlp_mult*dim² params.
            # SwiGLU uses 3 matrices of (dim × h): 3*h*dim params.
            # Equating: h = (2/3)*mlp_mult*dim. Round down to multiple of 64 for hardware alignment.
            hidden = max(64, (2 * mlp_mult * dim // 3 // 64) * 64)
            self.gate = CastedLinear(dim, hidden, bias=False)
            self.fc = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
            self.proj._zero_init = True
        else:
            hidden = mlp_mult * dim
            self.fc = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
            self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.use_swiglu:
            return self.proj(F.silu(self.gate(x)) * self.fc(x))
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SSMMixer(nn.Module):
    """Lightweight causal SSM-style mixer.

    This is not a full selective scan implementation, but a practical approximation
    suitable for quick A/Bs: projected channels + causal depthwise conv + gating.
    """

    def __init__(self, dim: int, expand: float = 2.0, kernel_size: int = 4):
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"SSM kernel must be >= 2, got {kernel_size}")
        hidden = max(64, int(dim * expand) // 64 * 64)
        self.in_proj = CastedLinear(dim, hidden * 2, bias=False)
        # Depthwise causal conv over time (implemented via left crop after padding).
        self.dw_conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            groups=hidden,
            bias=False,
            padding=kernel_size - 1,
        )
        self.out_proj = CastedLinear(hidden, dim, bias=False)
        self.out_proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        bsz, seqlen, _ = x.shape
        uv = self.in_proj(x)
        u, v = uv.chunk(2, dim=-1)
        u = F.silu(u)
        y = self.dw_conv(u.transpose(1, 2))[..., :seqlen].transpose(1, 2).contiguous()
        y = y * torch.sigmoid(v)
        return self.out_proj(y)


class MTPBranch(nn.Module):
    """Per-horizon residual branch for multi-token prediction."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm()
        self.proj = CastedLinear(dim, dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, h: Tensor) -> Tensor:
        return h + self.scale.to(dtype=h.dtype) * self.proj(self.norm(h))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        use_swiglu: bool = False,
        use_ssm: bool = False,
        ssm_expand: float = 2.0,
        ssm_kernel: int = 4,
    ):
        super().__init__()
        self.use_ssm = use_ssm
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        if use_ssm:
            self.attn = None
            self.ssm = SSMMixer(dim, expand=ssm_expand, kernel_size=ssm_kernel)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
            self.ssm = None
        self.mlp = MLP(dim, mlp_mult, use_swiglu=use_swiglu)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.use_ssm:
            if self.ssm is None:
                raise RuntimeError("SSM block is enabled but mixer is missing")
            mix_out = self.ssm(self.attn_norm(x))
        else:
            if self.attn is None:
                raise RuntimeError("Attention block is enabled but attention module is missing")
            mix_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * mix_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        recurrent_core_layers: int = 0,
        recurrent_steps: int = 0,
        share_ffn_across_blocks: bool = False,
        use_swiglu: bool = False,
        bigram_rank: int = 0,
        mtp_enabled: bool = False,
        mtp_steps: int = 2,
        mtp_weight: float = 0.3,
        mtp_decay: float = 1.0,
        mtp_tie_embeddings: bool = True,
        use_ssm: bool = False,
        ssm_every_n: int = 2,
        ssm_expand: float = 2.0,
        ssm_kernel: int = 4,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if (recurrent_core_layers > 0) != (recurrent_steps > 0):
            raise ValueError(
                "RECURRENT_CORE_LAYERS and RECURRENT_STEPS must both be > 0 for recurrence mode, "
                f"got RECURRENT_CORE_LAYERS={recurrent_core_layers}, RECURRENT_STEPS={recurrent_steps}"
            )
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.use_recurrence = recurrent_core_layers > 0 and recurrent_steps > 0
        self.recurrent_core_layers = recurrent_core_layers
        self.recurrent_steps = recurrent_steps
        self.share_ffn_across_blocks = share_ffn_across_blocks
        self.use_ssm = use_ssm
        self.ssm_every_n = ssm_every_n
        self.ssm_expand = ssm_expand
        self.ssm_kernel = ssm_kernel
        self.mtp_enabled = mtp_enabled and mtp_steps > 0
        self.mtp_steps = max(0, mtp_steps)
        self.mtp_weight = max(0.0, mtp_weight)
        self.mtp_decay = mtp_decay
        self.mtp_tie_embeddings = mtp_tie_embeddings
        self.total_effective_layers = (
            recurrent_core_layers * recurrent_steps if self.use_recurrence else num_layers
        )

        def is_ssm_block(idx: int) -> bool:
            return self.use_ssm and self.ssm_every_n > 0 and ((idx + 1) % self.ssm_every_n == 0)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        if self.use_recurrence:
            self.num_encoder_layers = 0
            self.num_decoder_layers = 0
            self.num_skip_weights = 0
            # In recurrence mode skip_weights are unused; keep as buffer so DDP
            # doesn't expect gradients for an empty parameter tensor.
            self.register_buffer("skip_weights", torch.ones(0, model_dim, dtype=torch.float32), persistent=False)
            self.blocks = nn.ModuleList(
                [
                    Block(
                        model_dim,
                        num_heads,
                        num_kv_heads,
                        mlp_mult,
                        rope_base,
                        qk_gain_init,
                        use_swiglu=use_swiglu,
                        use_ssm=is_ssm_block(i),
                        ssm_expand=ssm_expand,
                        ssm_kernel=ssm_kernel,
                    )
                    for i in range(recurrent_core_layers)
                ]
            )
            if share_ffn_across_blocks and len(self.blocks) > 1:
                shared_mlp = self.blocks[0].mlp
                for i in range(1, len(self.blocks)):
                    self.blocks[i].mlp = shared_mlp
        else:
            self.num_encoder_layers = num_layers // 2
            self.num_decoder_layers = num_layers - self.num_encoder_layers
            self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
            self.blocks = nn.ModuleList(
                [
                    Block(
                        model_dim,
                        num_heads,
                        num_kv_heads,
                        mlp_mult,
                        rope_base,
                        qk_gain_init,
                        use_swiglu=use_swiglu,
                        use_ssm=is_ssm_block(i),
                        ssm_expand=ssm_expand,
                        ssm_kernel=ssm_kernel,
                    )
                    for i in range(num_layers)
                ]
            )
        self.num_ssm_blocks = sum(1 for block in self.blocks if block.use_ssm)
        self.num_attn_blocks = len(self.blocks) - self.num_ssm_blocks
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if self.mtp_enabled:
            self.mtp_branches = nn.ModuleList([MTPBranch(model_dim) for _ in range(self.mtp_steps)])
            if self.mtp_tie_embeddings and self.tie_embeddings:
                self.mtp_heads = None
            else:
                self.mtp_heads = nn.ModuleList([CastedLinear(model_dim, vocab_size, bias=False) for _ in range(self.mtp_steps)])
            self.register_buffer(
                "mtp_step_weights",
                torch.tensor([self.mtp_decay**i for i in range(self.mtp_steps)], dtype=torch.float32),
                persistent=False,
            )
        else:
            self.mtp_branches = None
            self.mtp_heads = None
            self.register_buffer("mtp_step_weights", torch.zeros((0,), dtype=torch.float32), persistent=False)
        # Low-rank bigram logit bias.  At position i, adds bigram_right(bigram_left(input[i])) to logits.
        # This gives the model a cheap, learned n-gram prior on top of the contextual representations.
        self.bigram_rank = bigram_rank
        if bigram_rank > 0:
            self.bigram_left = nn.Embedding(vocab_size, bigram_rank)
            self.bigram_right = CastedLinear(bigram_rank, vocab_size, bias=False)
            self.bigram_right._zero_init = True   # starts contributing nothing; learns when useful
            self.bigram_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor, loss_mask: Tensor | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        if self.use_recurrence:
            for _ in range(self.recurrent_steps):
                for block in self.blocks:
                    x = block(x, x0)
        else:
            skips: list[Tensor] = []
            # First half stores skips; second half reuses them in reverse order.
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)

        h = self.final_norm(x)
        flat_h = h.reshape(-1, h.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(flat_h, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(flat_h)
        # Low-rank bigram bias: cheap learned n-gram prior on top of contextual representation.
        if self.bigram_rank > 0:
            bg = self.bigram_right(self.bigram_left(input_ids.reshape(-1)))  # [B*T, vocab]
            logits_proj = logits_proj + self.bigram_scale * bg
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        base_per_token = F.cross_entropy(logits.float(), targets, reduction="none")  # [B*T]
        if loss_mask is not None:
            mask = loss_mask.reshape(-1).to(base_per_token.dtype)
            base_loss = (base_per_token * mask).sum() / mask.sum().clamp(min=1)
        else:
            base_loss = base_per_token.mean()

        # Keep eval metric comparable by applying MTP only when loss_mask is not provided.
        if not self.mtp_enabled or self.mtp_weight <= 0.0 or loss_mask is not None:
            return base_loss

        _, seqlen = target_ids.shape
        weighted_aux = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        weight_sum = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        if self.mtp_branches is not None:
            for step_idx in range(self.mtp_steps):
                horizon = step_idx + 1  # 1 predicts token at t+2, 2 predicts t+3, ...
                if seqlen - horizon <= 0:
                    continue
                branch_h = self.mtp_branches[step_idx](h[:, : seqlen - horizon, :])
                branch_flat_h = branch_h.reshape(-1, branch_h.size(-1))
                future_targets = target_ids[:, horizon:].reshape(-1)
                if self.mtp_heads is None:
                    aux_logits_proj = F.linear(branch_flat_h, self.tok_emb.weight)
                else:
                    aux_logits_proj = self.mtp_heads[step_idx](branch_flat_h)
                aux_logits = self.logit_softcap * torch.tanh(aux_logits_proj / self.logit_softcap)
                aux_loss = F.cross_entropy(aux_logits.float(), future_targets, reduction="mean")
                w = self.mtp_step_weights[step_idx].to(dtype=weighted_aux.dtype)
                weighted_aux = weighted_aux + aux_loss.to(weighted_aux.dtype) * w
                weight_sum = weight_sum + w

        if torch.le(weight_sum, 0).item():
            return base_loss
        aux_loss = weighted_aux / weight_sum
        return base_loss + self.mtp_weight * aux_loss


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.quant_scheme not in SUPPORTED_QUANT_SCHEMES:
        raise ValueError(f"Unsupported QUANT_SCHEME={args.quant_scheme!r}; expected one of {sorted(SUPPORTED_QUANT_SCHEMES)}")
    if args.compressor not in SUPPORTED_COMPRESSORS:
        raise ValueError(f"Unsupported COMPRESSOR={args.compressor!r}; expected one of {sorted(SUPPORTED_COMPRESSORS)}")
    if args.weight_order not in SUPPORTED_WEIGHT_ORDERS:
        raise ValueError(f"Unsupported WEIGHT_ORDER={args.weight_order!r}; expected one of {sorted(SUPPORTED_WEIGHT_ORDERS)}")
    if args.mixed_low_precision_scheme not in {"int8", "int4"}:
        raise ValueError(
            f"Unsupported MIXED_LOW_PRECISION_SCHEME={args.mixed_low_precision_scheme!r}; expected 'int8' or 'int4'"
        )

    # -----------------------------
    # DISTRIBUTED + DEVICE SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_override = os.environ.get("DEVICE", "").strip().lower()
    grad_accum_override = os.environ.get("GRAD_ACCUM_STEPS", "").strip()
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if grad_accum_override:
        grad_accum_steps = int(grad_accum_override)
        if grad_accum_steps <= 0:
            raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}")
    else:
        if 8 % world_size != 0:
            raise ValueError(
                f"WORLD_SIZE={world_size} must divide 8 for default grad accumulation; "
                "set GRAD_ACCUM_STEPS explicitly to override"
            )
        grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    tokens_per_microstep = world_size * grad_accum_steps * args.train_seq_len
    if args.train_batch_tokens % tokens_per_microstep != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN; "
            f"got TRAIN_BATCH_TOKENS={args.train_batch_tokens}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if device_override:
        if device_override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested but CUDA is unavailable")
        if device_override not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported DEVICE={device_override!r}; expected 'cpu' or 'cuda'")
        device = torch.device(device_override, local_rank) if device_override == "cuda" else torch.device("cpu")
    else:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    autocast_enabled = device.type == "cuda"
    use_compile = bool(int(os.environ.get("USE_TORCH_COMPILE", "1" if device.type == "cuda" else "0")))
    if use_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    if distributed:
        if device.type == "cuda":
            dist.init_process_group(backend="nccl", device_id=device)
        else:
            dist.init_process_group(backend="gloo")
        dist.barrier()
    master_process = rank == 0

    sdp_backends_log = "cpu"
    if device.type == "cuda":
        # Fast math knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        # Some consumer GPUs and GQA configs do not support flash-only SDPA.
        # Default to "auto" so CUDA kernels can fall back to math/mem-efficient.
        sdp_backend_mode = os.environ.get("SDP_BACKEND_MODE", "auto").strip().lower()
        if sdp_backend_mode == "flash":
            enable_cudnn_sdp(False)
            enable_flash_sdp(True)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(False)
            sdp_backends_log = "cudnn=False flash=True mem_efficient=False math=False mode=flash"
        elif sdp_backend_mode == "math":
            enable_cudnn_sdp(False)
            enable_flash_sdp(False)
            enable_mem_efficient_sdp(False)
            enable_math_sdp(True)
            sdp_backends_log = "cudnn=False flash=False mem_efficient=False math=True mode=math"
        elif sdp_backend_mode == "auto":
            enable_cudnn_sdp(False)
            enable_flash_sdp(True)
            enable_mem_efficient_sdp(True)
            enable_math_sdp(True)
            sdp_backends_log = "cudnn=False flash=True mem_efficient=True math=True mode=auto"
        else:
            raise ValueError(
                f"Unsupported SDP_BACKEND_MODE={sdp_backend_mode!r}; expected 'auto', 'flash', or 'math'"
            )

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(f"device:{device} distributed:{distributed} use_torch_compile:{use_compile}", console=False)
    if device.type == "cuda":
        log0(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    if args.val_max_tokens > 0:
        usable = (min(args.val_max_tokens, val_tokens.numel() - 1) // args.train_seq_len) * args.train_seq_len
        if usable <= 0:
            raise ValueError(
                f"VAL_MAX_TOKENS={args.val_max_tokens} is too small for TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        val_tokens = val_tokens[: usable + 1].contiguous()
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1} "
        f"val_max_tokens:{args.val_max_tokens if args.val_max_tokens > 0 else 'full'}"
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        recurrent_core_layers=args.recurrent_core_layers,
        recurrent_steps=args.recurrent_steps,
        share_ffn_across_blocks=args.share_ffn_across_blocks,
        use_swiglu=args.use_swiglu,
        bigram_rank=args.bigram_rank,
        mtp_enabled=args.mtp_enabled,
        mtp_steps=args.mtp_steps,
        mtp_weight=args.mtp_weight,
        mtp_decay=args.mtp_decay,
        mtp_tie_embeddings=args.mtp_tie_embeddings,
        use_ssm=args.use_ssm,
        ssm_every_n=args.ssm_every_n,
        ssm_expand=args.ssm_expand,
        ssm_kernel=args.ssm_kernel,
    ).to(device=device, dtype=torch.bfloat16 if autocast_enabled else torch.float32)
    if autocast_enabled:
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=True) if use_compile else base_model
    model: nn.Module
    if distributed:
        model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if device.type == "cuda" else DDP(compiled_model, broadcast_buffers=False)
    else:
        model = compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=autocast_enabled,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=autocast_enabled,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if args.bigram_rank > 0:
        bigram_params = [base_model.bigram_left.weight, base_model.bigram_right.weight, base_model.bigram_scale]
        optimizer_bigram = torch.optim.Adam(
            [{"params": bigram_params, "lr": args.bigram_lr, "base_lr": args.bigram_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=autocast_enabled,
        )
        optimizers.append(optimizer_bigram)
    if args.mtp_enabled and base_model.mtp_branches is not None:
        mtp_params: list[nn.Parameter] = []
        for branch in base_model.mtp_branches:
            mtp_params.extend(list(branch.parameters()))
        if base_model.mtp_heads is not None:
            for head in base_model.mtp_heads:
                mtp_params.extend(list(head.parameters()))
        if mtp_params:
            optimizer_mtp = torch.optim.Adam(
                [{"params": mtp_params, "lr": args.mtp_lr, "base_lr": args.mtp_lr}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                fused=autocast_enabled,
            )
            optimizers.append(optimizer_mtp)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=autocast_enabled,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"sdp_backends:{sdp_backends_log}")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"use_swiglu:{args.use_swiglu} use_ssm:{args.use_ssm} ssm_every_n:{args.ssm_every_n} "
        f"ssm_expand:{args.ssm_expand} ssm_kernel:{args.ssm_kernel} "
        f"mtp_enabled:{args.mtp_enabled} mtp_steps:{args.mtp_steps} mtp_weight:{args.mtp_weight} "
        f"mtp_decay:{args.mtp_decay} mtp_tie_embeddings:{args.mtp_tie_embeddings} "
        f"qat_scheme:{args.qat_scheme} qat_start_step:{args.qat_start_step}"
    )
    if base_model.use_recurrence:
        log0(
            f"architecture:recurrent core_layers:{base_model.recurrent_core_layers} "
            f"recurrent_steps:{base_model.recurrent_steps} "
            f"effective_layers:{base_model.total_effective_layers} "
            f"ssm_blocks:{base_model.num_ssm_blocks} attn_blocks:{base_model.num_attn_blocks} "
            f"share_ffn_across_blocks:{base_model.share_ffn_across_blocks}"
        )
    else:
        log0(
            f"architecture:stacked num_layers:{args.num_layers} "
            f"encoder_layers:{base_model.num_encoder_layers} decoder_layers:{base_model.num_decoder_layers} "
            f"ssm_blocks:{base_model.num_ssm_blocks} attn_blocks:{base_model.num_attn_blocks}"
        )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} mtp_lr:{args.mtp_lr if args.mtp_enabled else 0.0}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(x, y)
                else:
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # SWA state: accumulated on CPU to avoid GPU memory pressure.
    swa_state: dict[str, torch.Tensor] | None = None
    swa_count = 0

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                autocast_enabled,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            # Load SWA-averaged weights before eval + export (better generalization + quantization).
            if args.swa_enabled and swa_state is not None:
                log0(f"swa: loading averaged weights from {swa_count} snapshots")
                base_model.load_state_dict(
                    {k: v.to(device=device, dtype=base_model.state_dict()[k].dtype) for k, v in swa_state.items()},
                    strict=True,
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # SWA: once warmdown begins (scale < 1), start averaging weights on CPU every N steps.
        if args.swa_enabled and scale < 1.0 and step % args.swa_collect_every == 0:
            if swa_state is None:
                swa_state = {k: v.detach().cpu().float().clone() for k, v in base_model.state_dict().items()}
                swa_count = 1
            else:
                inv = 1.0 / (swa_count + 1)
                for k, v in base_model.state_dict().items():
                    if k in swa_state:
                        swa_state[k].mul_(1.0 - inv).add_(v.detach().cpu().float(), alpha=inv)
                swa_count += 1

        # QAT: enable fake-quantisation once model has partially converged.
        # int8: single stage at qat_start_step (levels=256).
        # int4: 3-stage progressive schedule starting at qat_start_step:
        #   stage 0 (<33% of QAT window): levels=256  (gentle, int8-equivalent)
        #   stage 1 (33-67% of QAT window): levels=64
        #   stage 2 (>67% of QAT window): levels=16   (true int4)
        # Progressive avoids the catastrophic loss spike from jumping straight to 16 levels.
        if args.qat_scheme != "none":
            if step < args.qat_start_step:
                target_levels = 0
            elif args.qat_scheme == "int8":
                target_levels = 256
            else:  # int4 progressive
                qat_elapsed = step - args.qat_start_step
                qat_window = max(args.iterations - args.qat_start_step, 1)
                frac = qat_elapsed / qat_window
                target_levels = 256 if frac < 0.33 else (64 if frac < 0.67 else 16)
            if CastedLinear.qat_levels != target_levels:
                CastedLinear.qat_levels = target_levels
                log0(f"qat: {'enabled' if target_levels > 0 else 'disabled'} levels:{target_levels} step:{step}")

        # Sequence length curriculum: ramp from curriculum_min_seq_len → train_seq_len.
        if args.curriculum_enabled and step < args.curriculum_steps:
            frac_c = step / max(args.curriculum_steps, 1)
            curr_seq_len = args.curriculum_min_seq_len + int((args.train_seq_len - args.curriculum_min_seq_len) * frac_c)
            curr_seq_len = 1 << int(math.log2(max(64, curr_seq_len)))
        else:
            curr_seq_len = args.train_seq_len

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, curr_seq_len, grad_accum_steps)
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
            else:
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if device.type == "cuda":
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # a compressed quantized artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        raw_total_submission = model_bytes + code_bytes
        raw_budget_delta = args.submission_size_budget_bytes - raw_total_submission
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {raw_total_submission} bytes")
        if raw_budget_delta >= 0:
            log0(
                f"submission_budget raw_total:{raw_total_submission} budget:{args.submission_size_budget_bytes} "
                f"headroom_bytes:{raw_budget_delta}"
            )
        else:
            log0(
                f"submission_budget raw_total:{raw_total_submission} budget:{args.submission_size_budget_bytes} "
                f"over_bytes:{-raw_budget_delta}"
            )

    resolved_compressor, compressor_note = resolve_compressor(args.compressor)
    quant_obj, quant_stats = quantize_state_dict(
        base_model.state_dict(),
        scheme=args.quant_scheme,
        weight_order=args.weight_order,
        mixed_low_precision_scheme=args.mixed_low_precision_scheme,
    )
    artifact_name = export_artifact_name(args.quant_scheme, resolved_compressor)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_blob(quant_raw, resolved_compressor, args.compress_level)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_name)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["payload_bytes"], 1)
        if compressor_note:
            log0(f"export_note:{compressor_note}")
        log0(
            f"export_config quant_scheme:{args.quant_scheme} mixed_low_precision_scheme:{args.mixed_low_precision_scheme} "
            f"compressor:{resolved_compressor} weight_order:{args.weight_order} compress_level:{args.compress_level}"
        )
        log0(
            f"Serialized model {args.quant_scheme}+{resolved_compressor}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        quant_total_submission = quant_file_bytes + code_bytes
        quant_budget_delta = args.submission_size_budget_bytes - quant_total_submission
        log0(f"Total submission size {args.quant_scheme}+{resolved_compressor}: {quant_total_submission} bytes")
        if quant_budget_delta >= 0:
            log0(
                f"submission_budget {args.quant_scheme}+{resolved_compressor} total:{quant_total_submission} "
                f"budget:{args.submission_size_budget_bytes} headroom_bytes:{quant_budget_delta}"
            )
        else:
            log0(
                f"submission_budget {args.quant_scheme}+{resolved_compressor} total:{quant_total_submission} "
                f"budget:{args.submission_size_budget_bytes} over_bytes:{-quant_budget_delta}"
            )
        with open("final_export_manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "quant_scheme": args.quant_scheme,
                    "mixed_low_precision_scheme": args.mixed_low_precision_scheme,
                    "compressor_requested": args.compressor,
                    "compressor_resolved": resolved_compressor,
                    "compress_level": args.compress_level,
                    "weight_order": args.weight_order,
                    "artifact_name": artifact_name,
                    "artifact_bytes": quant_file_bytes,
                    "code_bytes": code_bytes,
                    "total_submission_bytes": quant_total_submission,
                    "submission_size_budget_bytes": args.submission_size_budget_bytes,
                    "budget_headroom_bytes": quant_budget_delta,
                    "baseline_tensor_bytes": quant_stats["baseline_tensor_bytes"],
                    "payload_bytes": quant_stats["payload_bytes"],
                    "raw_torch_bytes": quant_raw_bytes,
                    "payload_ratio": ratio,
                    "quant_format": quant_obj.get("__quant_format__", ""),
                },
                f,
                indent=2,
                sort_keys=True,
            )

    if args.final_roundtrip_eval:
        if distributed:
            dist.barrier()
        with open(artifact_name, "rb") as f:
            quant_blob_disk = f.read()
        quant_state = torch.load(
            io.BytesIO(decompress_blob(quant_blob_disk, resolved_compressor)),
            map_location="cpu",
            weights_only=True,
        )
        base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        q_val_loss, q_val_bpb = eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            autocast_enabled,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        roundtrip_tag = f"final_{args.quant_scheme}_{resolved_compressor}_roundtrip"
        log0(
            f"{roundtrip_tag} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
        )
        log0(f"{roundtrip_tag}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    else:
        log0("final_roundtrip skipped FINAL_ROUNDTRIP_EVAL=0")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
