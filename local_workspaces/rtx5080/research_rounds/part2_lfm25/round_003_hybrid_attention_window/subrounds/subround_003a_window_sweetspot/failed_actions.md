# Subround 003a Failed Actions

## 2026-07-19 - Normal client startup exceeded 4 GiB

The first build-only tokenizer probe used the R002k image's normal Python
startup and exited `137` under a 4 GiB client limit. The image-level
`sitecustomize` imported the vLLM research hook before the tokenizer client.
This did not exercise the model or invalidate any server measurement. The
client now runs with `python3 -S` and adds only the image's `purelib` path
inside the script, preventing server hooks from loading in the benchmark
client.

Do not treat the short GPQA proxy as evidence that a long-context attention
window is semantically safe. Its maximum prompt length is below every window
in this sweep.

## 2026-07-19 - First retrieval calibration did not reach window boundaries

The initial pure-stdlib probe estimated four characters per token. The server
reported only `1,835..13,320` tokens for labels `3,072..22,528`, so that run
was a tokenizer calibration rather than a semantic comparison. The filler
estimate is now `6.7` characters per token and every result records the exact
server-reported prompt length.

## 2026-07-19 - One 6144 profile process was contaminated

The discarded `6144-B` process reported about `81.9ms` at turn 0, where a
6144-token window is not active. That is roughly 20ms above both neighboring
valid processes and cannot be caused by truncating long attention history.
It is excluded before aggregation; valid A/C processes remain in the compact
profile result. Do not average a process-wide slowdown into a window-size
effect.

## 2026-07-19 - Profile savings did not predict 12288 full score

Window `12288` passed the isolated turn-5 GPU gate with a `27.294ms` saving,
but its only full replay scored `57.0213`, only `+0.4763` over the repeated
unbounded mean and far below `8192`. Single-conversation GPU timing is an
opening gate, not a score proxy; admission, batching and decode interactions
must be measured on the full trace.

## 2026-07-19 - Window 4096 failed the paired retrieval gate

Two independent full replays make `4096` the stable local performance winner,
but it loses two passkey cases that the same unbounded control answers at
approximately 5.1k and 7.1k prompt tokens. Short GPQA cannot detect this
because its inputs are below 1k tokens. Do not label `4096` quality-safe or
promote it solely from its `64.5504` mean score.
