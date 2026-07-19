# R002y Results - Cached-Turn Prefill Attribution

Status: complete. Attribution passed; no serving candidate was promoted.

## Valid Bracket

The final bracket used three clean R002k processes:

```text
control A: final_control_a-timing-20260719-181415, four repeats/turn
Kineto:   final_attribution-kineto-20260719-181527, one trace/turn
control B: final_control_b-timing-20260719-181646, four repeats/turn
```

Each process ran all 90 public primer requests unarmed before measurement.
Every measured panel contained one isolated request, reached one output token
and recorded the expected cache progression. The complete-GPU A/B weighted
drift was `1.418%`, below the registered `10%` limit.

## Turn Progression

```text
turn  prompt   cached before  new prefill  cache hit  complete GPU mean
0      4000        0             4000        0.00%        60.801 ms
1      8199     4000             4199       48.79%        75.114 ms
2     12398     8192             4206       66.08%        85.242 ms
3     16597    12384             4213       74.62%        95.948 ms
4     20796    16592             4204       79.78%       105.430 ms
5     24994    20784             4210       83.16%       114.685 ms
```

Prefix caching is working: after turn 0, newly scheduled work remains near
`4.2k` tokens even as total prompt length reaches `24,994`. It does not make
cached-turn attention constant-time. Each new query still attends the retained
KV context in the six full-attention layers.

The first-token GPU boundary grows `53.884ms` from turn 0 to turn 5. Attention
grows `49.917ms` over the same interval, explaining `92.64%` of that growth.
FP8 GEMM and BF16 GEMM remain approximately flat across turns.

## Child-Kernel Attribution

Equal turn weights match the public trace's 55 graded requests per turn index:

```text
family             weighted ms    share      present turns
FP8 GEMM              39.713      45.87%          6/6
attention             30.029      34.69%          6/6
BF16 GEMM             13.930      16.09%          6/6
all remaining          2.897       3.35%          6/6
```

FP8 GEMM is the largest fixed cost, but attention is the only family that
tracks the cached-context growth. At turn 5, six local FlashInfer paged-prefill
kernels consume `54.527ms` of `111.777ms` child-kernel time.

Turns 1-5 also split at the aligned Mamba boundary. A large first step schedules
`4192-4208` tokens and a second full model step schedules only `2-14` tokens.
The second steps cost `2.662-2.923ms`, or `2.296ms` per turn when weighted over
all six turns. This is a real secondary opportunity, but removing it requires
preserving an intermediate aligned Mamba state and is not a safe scalar flag.

## H200 Portability

The local kernel name is not the target kernel name. Exact vLLM `0.22.1`
source and image binaries establish:

- RTX 5080 SM120 rejects FlashAttention for FP8 KV and falls through to
  FlashInfer; the measured local prefill kernel is therefore FlashInfer.
- H200 SM90 has the packaged FA3 binary, FA3 explicitly supports FP8 E4M3 KV
  on capability family 90, and automatic backend priority selects
  FlashAttention before FlashInfer.
- TRT-LLM attention in this image requires SM100 and cannot be an H200 path.
- `use_prefill_decode_attention` is declared in `AttentionConfig` but has no
  runtime call site in this vLLM build.

The architectural conclusion transfers: full attention scans a growing cached
context. The local FlashInfer duration and any local FlashInfer replacement
speedup do not transfer numerically to H200 FA3. See `source_audit.md`.

## Gate And Decision

The measurement gate passes: all turns are valid, cache progression is valid,
control drift is low, and attention contributes `30.029ms`/`34.69%` across all
six turns. The implementation gate remains intentionally closed because the
measured local backend differs from H200 and no proven prefill-only change can
yet preserve the incumbent decode path.

Decision:

- complete R002y as causal attribution evidence;
- keep R002k and its decode policy unchanged;
- create no overlay, quality run, image, Compose or Portal attempt;
- open only an H200-specific FA3 prefill qualification round, with an explicit
  source/runtime gate before any submission;
- retain aligned-state tail elimination as a secondary `2.296ms` ideal ceiling,
  not an approved implementation.
