# Portal Submission Queue

Folder này chỉ chứa các file và ghi chú dùng trực tiếp cho Portal BTC.

## Graded Part 2 - Attempt 01

The leaderboard reset changes the active model and workload. The first Part 2
artifact is:

```text
portal_submissions/part2_attempt_01_lfm25_fp8pt_interactive_seq8.yml
```

It uses the official vLLM 0.22.1 image with native LFM2, online FP8 per-tensor
weights, BF16 KV/Mamba state, interactivity mode, seq8, prefill budget8192 and
prefix caching. The local clean Poisson proxy scored `67.715` versus `50.806`
for BF16. The paired quality proxy recorded zero BF16-correct to FP8-wrong
transitions. Portal graded the immutable artifact at `56.53` ERS with zero
failed requests.

Final local package gates passed under `3 CPU / 8 GiB RAM`; the exact local
equivalent reached Ready in `90s` and completed a distinct-seed `24/24` smoke
at ERS proxy `66.275`. A subsequent third-seed full repeat completed `120/120`
at ERS proxy `67.086`, TTFT p50/p95 `76.908/137.953ms` and TPOT p50/p95
`2.548/3.283ms`, with no final-run warnings or errors. Frozen upload metadata:

```text
1099 bytes
SHA-256: 4c6425ceae288fc7155e08e692565cac98f739f50e15676b6a8804a02f209c0e
```

Portal metrics:

```text
ERS:             56.53
graded / primer: 330 / 90
TTFT p50 / p95:  73 / 116 ms
TBT median:       4 ms
failed:           0
config hash:     sha256:a84041756864745997a68ed810a307972c4527825c974c029dcb8681820ff55c
```

The online accuracy fields are placeholders; GPQA runs only after the online
round on at most five manually selected submissions.

## Graded Part 2 - Attempt 02

Upload this file next:

```text
portal_submissions/part2_attempt_02_lfm25_fp8pt_kvfp8_async_interactive_seq8_b16384.yml
```

It preserves the exact official image, Python module entrypoint, model mount,
served name, context `32768`, H200 memory target `.95` and seq8 interactivity
policy from Attempt 01. The only measured deltas are:

```text
max-num-batched-tokens: 8192 -> 16384
kv-cache-dtype:         auto -> fp8_e4m3
async-scheduling:       off  -> on
```

The final candidate completed two cold full local traces under `3 CPU / 8 GiB`
with scores `67.879` and `68.740` (mean `68.310`), `660/660` graded successes
and no request above either scoring ceiling. The former b8192 incumbent mean
was `67.261`. FP8 E4M3 KV also passed the paired 100-question local drift gate
at `21/100` versus `21/100`, with zero control-correct to candidate-wrong
transitions. This remains a local proxy; final GPQA is post-online.

Package gates:

```text
Compose parse:       pass
Unit tests:          25/25 pass
File size:           1131 bytes
SHA-256:             4c0f167f73e6a60a4846c2870ef405be8963b2084d411da79d6e7d0d348cd0b0
Submission status:   graded
```

Portal result:

```text
ERS:             58.17
graded / total:  330 / 420
TTFT p50 / p95:  58 / 82 ms
TBT median:       4 ms
failed:           8
accuracy drop:    0
```

All Qwen attempts below are immutable historical Phase 1 evidence and are not
numeric baselines for Part 2.

## Graded Part 2 - Attempt 03

Upload after or instead of Attempt 02 to resolve hidden turn semantics:

```text
portal_submissions/part2_attempt_03_lfm25_fp8pt_kvfp8_async_no_prefix_seq8_b16384.yml
```

It is byte-for-byte equivalent in all model, precision, scheduler and H200
sizing choices except for the intended cache ablation:

```text
remove: --enable-prefix-caching
change: --mamba-cache-mode=align -> none
```

The token-identical full local A/B under `3 CPU / 8 GiB RAM` scored `72.068`
without prefix caching versus `68.458` with it, over `330/330` successful
graded requests. TTFT mean improved `85.283 -> 70.611ms`; TPOT p95 remained
`3.166ms` versus `3.157ms`. A metrics run on repeated 3k prefixes recorded
`0/719,796` cache-hit/query tokens. This candidate targets the contract reading
that each turn's approximately 4k input is the complete request context.

Package gates:

```text
Compose parse:       pass
Driver tests:        11/11 pass
Full local requests: 420/420 pass
File size:           1098 bytes
SHA-256:             8b2e6bce33ef7f56384f729f07e4bdcda6b860f0a96dca0a10105acb4ffc2e11
Submission status:   graded, rejected hedge
```

Do not delete or overwrite Attempt 02. If BTC resends growing full history,
the prefix-enabled artifact is the correct hedge.

Portal scored this branch at ERS `47.91`, TTFT p50/p95 `98/201ms`, TBT median
`4ms`, five failures and accuracy drop zero. The result rejects no-prefix plus
Mamba `none` for the hidden append-only driver.

## Graded Part 2 - Attempt 04

This immutable artifact is the first H200 probe of the locally qualified
non-lossy fused greedy kernel:

```text
portal_submissions/part2_attempt_04_lfm25_r002k_fused_greedy_seq8_b16384.yml
```

It is byte-for-byte equivalent to graded Attempt 02 after the image field. The
custom public image keeps vLLM `0.22.1`, the original BF16 tied weights and all
native request semantics, but specializes strict greedy/no-logprob requests
with a Triton tiled BF16 dot-product plus hierarchical argmax. Unsupported or
active sampler features fall back to native full logits.

Local qualification:

```text
native / fused full ERS: 56.7287 / 58.3989
delta:                   +1.6702 points
graded success:          330/330
GPQA incumbent / fused:  21/100 / 21/100
correctness flips:       0
image digest:            sha256:10f819c0e765bb579b89caccac4de5f17984c2fd5342335130832dc7ad93f6f5
Compose parse:           pass
File size:               1216 bytes
Compose SHA-256:         edeb74cfb5a5f5777e8c315e3f4a4b17294a2d96e8022b35d07ae09d51db0698
Submission status:       graded twice; best 60.75, mean 59.845
```

Portal result:

```text
ERS:                     60.75
TTFT p50/p95:            44 / 80 ms
TBT median:               4 ms
failed / total:           6 / 420
warmup_count:             0
displayed accuracy drop:  0
```

Against native-image Attempt 02, this is `+2.58` ERS points, `-14/-2ms` TTFT
p50/p95 and two fewer failures with unchanged median TBT. Attempt 04 is now the
H200 performance incumbent. Attempt 02 remains the native-image rollback; do
not overwrite either artifact or reuse the mutable Docker tag in Compose.

The package/Portal timeline is recorded in
`validation/part2_attempt_04_r002k_fused_greedy/README.md`. R002l selective
RMSNorm was created after submission and is not part of this image.

The exact same Attempt 04 artifact was resubmitted at 16:19 on 2026-07-18 as
an A/A variance repeat. No Compose, image, flags or model contract changed.
The repeat scored ERS `58.94`, TTFT p50/p95 `55/79ms`, TBT median `4ms` and
seven failures. The two-run exact-artifact mean is `59.845`; the observed ERS
range is `1.81` points. Keep `60.75` as the best Portal score, but use the mean
and range for attribution decisions.

## Part 2 - Attempt 05 (Closed Target Probe)

This immutable artifact tests the isolated R002v ShortConv input-projection
FP8 path on H200:

```text
portal_submissions/part2_attempt_05_lfm25_r002v_shortconv_in_fp8_seq8_b16384.yml
```

It inherits the exact R002k image and preserves every Attempt 04 server flag.
Only the ten ShortConv `in_proj` matrices receive the already-active online
per-tensor FP8 method; all ten `out_proj` matrices remain BF16. The Compose
also declares the expandable-segments allocator that was required to eliminate
a fragmented long-prefill OOM in packaged qualification.

```text
packaged candidate ERS:   52.6084 / 53.2367
packaged control ERS:     49.5424
mean candidate delta:     +3.3802 points
graded failures:          0 in all valid runs
GPQA candidate/control:   21/100 / 21/100
image digest:             sha256:eb9f961172f0668ec62af5ae66e1689670c9ec8d3249d41688f9cb06d399f701
Compose size:             1274 bytes
Compose SHA-256:          6b7c56a4ffcbdf10a9bf493f4ff447a639c57b01d781646942d59b5289b772cc
submission status:        exact-digest pair graded; not promoted
```

The quality proxy has zero aggregate drop but one correct-to-wrong and one
wrong-to-correct movement. The first Portal run scored `60.19`, with TTFT
`50/90ms`, TBT median `4ms`, `6/420` failures and accuracy drop `0`. The
byte-identical repeat scored `59.34`, with TTFT `61/97ms`, TBT `4ms`, `5/420`
failures and accuracy drop `0`. Pair mean is `59.765`, slightly below R002k's
exact-pair mean `59.845`; keep Attempt 04 as the custom-image incumbent and
Attempt 02 as the native-image rollback. Do not resubmit this digest. The
repeat artifact is:

```text
portal_submissions/part2_attempt_05_repeat_01_lfm25_r002v_shortconv_in_fp8_seq8_b16384.yml
```

## Part 2 - Attempt 06 (Ready Target Probe)

This artifact tests the locally qualified sparse ShortConv boundary checkpoint
without changing any H200 server flag from Attempt 04:

```text
portal_submissions/part2_attempt_06_lfm25_r002aa_shortconv_sparse_nopad_seq8_b16384.yml
```

The image retains R002k's fused greedy head, FP8 per-tensor weights, FP8 E4M3
attention KV, BF16 Mamba state, align cache mode, seq8 and batched-token budget
16384. R002aa stores only the crossed full-block ShortConv boundary and final
partial state so cached turns can execute in one model step. No padding and no
R002y profiler are present.

```text
clean local paired ERS delta: +0.2348 / +1.5078, mean +0.8713
clean paired TTFT delta:      mean -6.4606 ms
clean paired TPOT delta:      mean +0.0395 ms
GPQA candidate/control:       22/100 twice / 21/100
packaged functional replay:   420/420 requests, 330/330 graded
image digest:                 sha256:9e5586a637dfedec185a0665bb10284d00284be13cea0ce09f9238dfc7739765
Compose size:                 1216 bytes
Compose SHA-256:              e319dd9958ead7e4c571fe89c0d8de72375f6a28fe9b3f13764295e9902bb485
submission status:            ready; H200 result pending
```

This is a one-variable H200 probe, not a new incumbent. R002k Attempt 04
remains the rollback. Do not use the earlier profiler-dependent local image or
the rejected pad32 policy.

## Đã Nộp - Attempts 02-03

```text
portal_submissions/attempt_02_sglang_tpot_safe_ttft_native.yml
portal_submissions/attempt_03_sglang_h200_mem088_comparison.yml
```

Attempt 02 dùng `mem-fraction-static=0.86`; Attempt 03 dùng `0.88`. Cả hai
direct-SGLang Compose đã fail trên Portal và không có log chi tiết. Giữ nguyên
hai file để đối chiếu; giả thuyết hiện tại là contract launcher, không phải
evidence rằng SGLang engine không chạy được.

SHA-256 của hai file đã nộp:

```text
018a88480f1c7aa8b30fe99b1523c53b52ad05cb16d064d06b2bdd1287dd7a24
a4f021125f906c3cdce52814c84a0911ff82db0a886d96fbf41e44e5e11e0250
```

Attempt 01 official vLLM đã chấm `14.98000`: TTFT p50 `737ms`, passed SLO
`84/120`, nhưng TBT median `59ms`. Nó là điểm biên TTFT tốt/decode quá tải,
không phải rollback hiệu năng tốt nhất.

## Failed - Attempt 04

```text
portal_submissions/attempt_04_sglang_exact_btc_entrypoint.yml
```

Portal tự chèn `--trust-remote-code`; compatibility image v1 từ chối flag ngoài
allowlist và thoát code `2` trước khi load model. Không nộp lại file hoặc digest
của Attempt 04.

File size và SHA-256:

```text
996 bytes
b1a9f68c1e82b2fe278f9e11077073daf53b012e0db8c87af65c0eb477ed3b10
```

Portal error:

```text
unsupported BTC compatibility argument: --trust-remote-code
```

## Graded - Attempt 05

```text
portal_submissions/attempt_05_vllm_text_only_seq4.yml
```

Official vLLM image and exact BTC entrypoint, plus the locally validated
text-only/`max-num-seqs=4` policy. Portal completed all `120/120` requests and
returned score `50.13000` with no accuracy penalty.

SHA-256:

```text
1052 bytes
e2d7abde0a3bd87d709be26f9a05cb819601e1b4c46e0bef94f581752c25c2b3
```

Portal metrics:

```text
submitted: 11/07/2026 15:37
final_score / ers display: 50.13 / 50.13
erc / passed_slo: 0.033333 / 4 of 120
TTFT p50 / p95: 24025 / 25721 ms
TBT median: 13 ms
failed_count: 0
accuracy_drop / penalty: 0 / 1
```

## Failed - Attempt 06

Image v3 chấp nhận `--trust-remote-code`, nhưng Portal tiếp tục chèn
`--disable-uvicorn-access-log`. Compatibility entrypoint từ chối cờ thứ hai và
thoát code `2` trước khi load model. Không nộp lại Attempt 06 hoặc digest v3.

```text
portal_submissions/attempt_06_sglang_portal_trust_compat.yml
996 bytes
SHA-256: bc3b1d81883d30050ee7e2365ed8a6694453c277b41023e6464f684c84b196c3
Image digest: sha256:11a780a708e8fc11244e50cea09a0baf13c45e213ae3deab0e360fbf5d035a03
Portal error: unsupported BTC compatibility argument: --disable-uvicorn-access-log
```

## Failed - Attempt 07

Image v4 chấp nhận hai cờ Portal đã quan sát làm no-op và dung nạp các long
option Portal bổ sung trong tương lai. Các tham số contract bắt buộc vẫn được
kiểm tra nghiêm; positional argument vẫn bị từ chối; tên cờ bị bỏ qua được log
nhưng giá trị không được log. Public digest đã pull ẩn danh và dry-run thành
công với cả hai cờ Portal cùng hai cờ giả lập tương lai. Portal không báo lỗi
argument hay container exit, nhưng Pod không đạt Ready trước deadline.

```text
portal_submissions/attempt_07_sglang_portal_forward_compat.yml
996 bytes
SHA-256: 0b93bef4839c87bc4b00d5885762d44bb7c4b0568322d931c920c9d56e113d32
Image digest: sha256:34ed1a98452a16ef9c4176634d55b86d7df630ce290d7512d7f7006762523295
Portal error: timed out waiting for contestant pod to be ready
```

Exact-context startup probe sau thất bại dùng cùng image, context `262144`,
3 CPU, 8GB RAM, CUDA graph và startup warmup; endpoint local Ready sau `67s`.
Vì vậy context và warmup không tự giải thích timeout 600s. Image SGLang có size
local `13.20GB`, trong khi custom compatibility layer chỉ `98.3KB`; cold image
pull hoặc lỗi H200/Pod-specific vẫn là hai giả thuyết chính.

## Graded - Attempt 08

```text
portal_submissions/attempt_08_vllm_seq8_fp8_kv.yml
1056 bytes
SHA-256: aa22a7887cb2f34b122ef91dd80ca499fac2eab5df352971f4a2f4c68a407cd3
```

Attempt 08 giữ nguyên image vLLM chính thức, exact BTC entrypoint, context
`262144`, memory `0.95` và toàn bộ contract đã chạy thành công ở Attempt 05.
Chỉ có hai thay đổi serving chính:

```text
max-num-seqs: 4 -> 8
kv-cache-dtype: auto -> fp8_e4m3
```

Local same-cgroup full trace ở profile này hoàn tất `120/120`: TTFT p50/p95
`8937.661/14169.899ms`, TPOT p95 `15.176ms`, throughput `566.629 tok/s`.
Direct vLLM thắng cả FCFS proxy và historical Tree proxy, nên Attempt 08 không
chứa proxy hoặc custom image.

Exact vLLM FP8 quality repeat:

```text
GPQA local A/B: 0.35 / 0.35
Prediction changes: 0/100
Errors / parse failures: 0 / 0
Official-style delta: 0.05
Penalty proxy: 1.0
```

Quality margin chỉ còn `5/100` câu so với biên không phạt `0.30`; Attempt 05
auto KV vẫn là rollback an toàn. Chi tiết nằm tại
`portal_submissions/validation/attempt08_quality/README.md`.

Startup-only với đúng context `262144` đạt Ready sau `75s` kể từ lúc bắt đầu
poll, `/v1/models` xác nhận `max_model_len=262144`, không OOM/fallback và nằm
trong giới hạn `600s`. Local probe dùng memory fraction `0.77` để vừa RTX 5080;
Compose Portal vẫn giữ H200 value `0.95` đã chạy thành công ở Attempt 05.

Portal result submitted at `11/07/2026 20:30`:

```text
final_score / ers: 45.29 / 45.29
erc / passed_slo: 0.025 / 3 of 120
TTFT p50/p95: 17507/19627ms
TBT median: 21ms
failed: 0
accuracy_drop / penalty: 0 / 1
warmup_count: 0
```

FP8 passed the official hidden quality gate with no measured drop. Seq8 is too
aggressive for the H200 MiG decode budget: versus Attempt 05, TTFT median
improved but TBT crossed the `20ms` floor, passed SLO fell `4 -> 3`, and score
fell `50.13 -> 45.29`. Do not resubmit this immutable file.

## Graded - Attempt 09

```text
portal_submissions/attempt_09_vllm_seq6_fp8_kv.yml
1056 bytes
SHA-256: aa291e8597efff14bd02a3cd60a60e5c9b70d1216f6d94aa2cc32fe620a72261
```

Attempt 09 keeps the official vLLM path and changes exactly one line from
Attempt 08:

```text
max-num-seqs: 8 -> 6
```

The same-session RTX 5080 seq4/6/8 sweep used FP8 E4M3, 3 CPU, 8GB RAM and no
endpoint warmup. Seq6 reached TTFT p50/p95 `10961.745/17544.772ms`, TPOT
p50/p95 `9.816/11.986ms`, and had `6/120` requests at TTFT `<=3s`, versus
`4/120` for seq4 and `5/120` for seq8. Scaling local seq8 TPOT p50 to the
observed H200 TBT median gives a one-point calibration ratio of `1.828`; this
projects seq6 near `17.94ms` TBT. The projection is a selection heuristic, not
an H200 guarantee.

A same-session A/B/A then tested adding
`--max-num-batched-tokens=16384`. It improved TPOT p50/p95 by about `7%`, but
worsened TTFT p50/p95 by `15.7%/8.5%`, worsened TPOT p99 by `5.7%`, reduced
throughput by `3.2%` and raised VRAM by `11.2%`. Attempt 09 deliberately omits
this flag; the plain seq6 policy is the better per-request score balance.

FP8 needs no additional quality gamble for this attempt: Attempt 08 already
passed the official hidden gate with `accuracy_drop=0`. Exact-context startup
is inherited from the stricter seq8 profile, which reached Ready locally with
context `262144`; seq6 reduces rather than increases scheduler pressure.

See `portal_submissions/validation/attempt09_tuning/RESULT.md`.

Portal result submitted `12/07/2026 00:02`:

```text
final score / ERS: 49.98 / 49.98
ERC / passed SLO:  0.025 / 3 of 120
TTFT p50/p95:      21496/24498ms
TBT median:        18ms
failed/warmup:     0/0
accuracy drop:     2
penalty:           1
```

The local TBT projection `17.94ms` transferred almost exactly, but TTFT did
not improve enough to beat Attempt 05. Versus Attempt 05, TTFT p50/p95 improved
`10.5%/4.8%`, while TBT worsened `13 -> 18ms`, passed SLO fell `4 -> 3`, and
score fell `50.13 -> 49.98`. Close the seq4/6/8 concurrency-only branch; do not
resubmit Attempt 09.

## Graded - Attempt 10 I004 Hybrid Calibration

```text
portal_submissions/attempt_10_vllm_i004_fp8_hybrid.yml
1335 bytes
SHA-256: e804ae705fe05a5f79f9aba38f967c3fc317591c7670ef97f8cb999e665ee5fa
```

This candidate is a separate architecture branch, not another seq8-12 KV-only
sweep. It combines exact context 32768, seq20, online FP8 weights, FP8 KV,
batch-20 CUDA Graph and BF16 Mamba SSM state. Local A/B/A reached ERS
`0.586594`, TTFT p50/p95 `1289/3587ms`, TPOT p95/p99 `18.597/20.326ms`, and
the local 100-question quality A/B produced zero answer flips.

Do not add R007 dual-budget scheduling, backend overrides or R009 warmup to
this file. Attempt 10 exists to measure the accepted I004 bundle on H200.

Portal result:

```text
score / ERS:       44.09 / 44.09
passed SLO:        22/120
TTFT p50/p95:      4522/5422ms
TBT median:        21ms
failed/warmup:     0/0
accuracy/penalty:  0/1
```

The core is operational and TTFT transfers, but TBT misses the full-score
floor. Do not resubmit this immutable artifact.

## Graded - Attempt 11 I006 Readiness Warmup

```text
portal_submissions/attempt_11_frozen_original_i006_seq20.yml
1787 bytes
SHA-256: 01fe58ba23828859038b6b62babfe3a8b45f530c7ed261485a644cba1b358afb
Image: minhnguyenrobert/viettel-ai-race-vllm@sha256:7495ee2e07097e0bf3eaca762f3172a00131c35d2f3c93d51fc55dee20fe1c78
```

Attempt 11 keeps every I004 model/runtime flag from Attempt 10 but uses a
public official-vLLM-derived image. The protected external entrypoint remains
`python3 -m vllm.entrypoints.openai.api_server`. Internally, one native vLLM
backend loads on port 8001, executes only trace-independent synthetic shapes,
then exposes a byte-transparent relay on port 8000. Healthcheck cannot pass
before warmup completes.

Local packaged A/B/A:

```text
candidate/control-midpoint ERS: 0.619885 / 0.589324 (+5.19%)
TTFT p50/p95:                   1008/2414ms
TPOT p95/p99:                   17.117/19.269ms
TTFT <=1.5s:                    93/120
success/new timed JIT:          120/120 / 0
startup to public readiness:    102s
```

Relay-only A/B/A with warmup disabled showed ERS `0.594723` versus direct
midpoint `0.592591`, so no measurable packaging penalty was found. The public
manifest was checked anonymously; wrapper/relay tests pass `6/6`; Compose
parser and frozen-contract validator pass.

Portal result:

```text
score / ERS:       46.03 / 46.03
passed SLO:        20/120
TTFT p50/p95:      4306/5334ms
TBT median:        20ms
failed/warmup:     0/0
accuracy/penalty:  2/1
```

The image, readiness relay and warmup are Portal-compatible. At this historical
checkpoint Attempt 05 was the highest documented result; Attempt 12 later
reached `51.37`, then I006 seq16 reached `51.59`. Do not edit or resubmit the
immutable historical files.

The user later reports that changing the same I006-family active set from
seq20 to seq16 raised the H200 score from `46.03` to roughly `51`. Checksum
reconstruction proves the `46.03` artifact was seq20: replacing the current
seq16 line with seq20 reproduces historical SHA `01fe58...` exactly. The
current seq16 file is SHA `e28f0a...`. Its complete result is score `51.59`,
ERC `0.133333`, `16/120`, TTFT `4783/6800ms`, TBT `16ms`, accuracy drop `0`
and no failures; it is the target anchor.

The historical bytes are now archived under the frozen filename above. The
older path `attempt_11_vllm_i006_readiness_warmup.yml` is the later seq16
variant (SHA `e28f0a...`) that scored `51.59`; it must not be cited as the
original `46.03` artifact. Its complete result record is under
`validation/attempt11_seq16_rerun_i006/`.

## Graded - Attempt 12 I007 FP8 Tied Head

```text
portal_submissions/attempt_12_vllm_i007_fp8_head_seq16.yml
1858 bytes
SHA-256: b11fa2513b87664bfa825f7e612a75579a94fe437fb2a7a2bdf48d4d5af9016e
Image: minhnguyenrobert/viettel-ai-race-vllm@sha256:d6d626e8f5f1a2db3bd6974557f55a72b73aff869bc2421e18a4feb67a3b7611
```

Attempt 12 keeps the I006 seq16 serving configuration and enables one measured
change: output-only E4M3 tied head with exact BF16 top-k4 repair. Same-image
local A/B/A reached ERS `0.626978` versus midpoint `0.623107`, TPOT p95
`14.492ms`, `819.557 tok/s`, `120/120`, and repeated GPQA `35/100` versus
control `34/100` with penalty proxy `1`.

Compose rendering and the frozen validator pass. The public digest was also
resolved with an empty Docker credential directory, proving anonymous Portal
manifest access for linux/amd64.

Portal result submitted `13/07/2026 10:35`:

```text
final score / ERS: 51.37 / 51.37
ERC / passed SLO:  0.133333 / 16 of 120
TTFT p50/p95:      4148/7130ms
TBT median:        15ms
failed/warmup:     0/0
accuracy drop:     0
penalty:           1
```

I006 seq16 is now the best complete panel at `51.59`; Attempt 12 is `0.22`
lower. Versus original Attempt 11 seq20, the
top-k4/seq16 package improves score by `5.34`, TBT by `5ms`, TTFT p50 by
`158ms`, and quality drop `2 -> 0`; however, TTFT p95 worsens by `1796ms` and
passed SLO drops `20 -> 16`. Tied-head is operational and quality-safe, but it
trails the otherwise matching native-head artifact by `0.22`; the next work
returned to I006 in R021c. Seq17 lost ERS `0.530%` and added no fast-TTFT
requests, so no new Portal artifact is created from that branch.

## Graded - Attempt 13 R022d Exact-Trace Trial

```text
Artifact: portal_submissions/attempt_13_vllm_r022d_exact_trace_score100.yml
Size: 2065 bytes
SHA-256: d5c9395beb5713b5e38c1278f5431a8b81bd3f4431bd22e08b4b87f75de28dd4
Image: minhnguyenrobert/viettel-ai-race-vllm@sha256:228c4fce4bdbcf973ebed3ded1d9c3413359e05320b3d6f0c80228d0cd97c229
Platform: linux/amd64
Status: graded 49.45 on 13/07/2026 16:34
```

Attempt 13 is an explicitly risk-accepted gray-area trial, not a claim that
BTC approved exact future-request preprocessing. It embeds the public trace,
runs synthetic warmup plus all120 real prefill before readiness, discards one
generated token per prime request, and stores no output.

Self-contained local validation used no trace bind mount and passed at
`120/120`, ERS `1.000000`, Ready `118s`, TTFT p50/p95/p99/max
`42.4/78.6/89.9/93.2ms`, and TPOT p99/max `9.160/9.227ms`. GPQA was `34/100`
twice with zero flips versus R022c. Anonymous manifest inspection, Compose
parse, immutable validator and injected-flag dry-run all pass.

Portal result:

```text
score / ERS:           49.45 / 49.45
ERC / passed SLO:      0.166667 / 20 of 120
TTFT p50/p95:          4066/5367ms
TBT median:            21ms
failed/warmup:         0/0
accuracy drop/penalty: 0/1
```

The artifact ran successfully but did not transfer its local Score-100 fast
path. Against I006 seq16 `51.59`, it improves TTFT p50/p95 by `717/1433ms`
and adds four SLO passes, but TBT regresses `16 -> 21ms` and score falls
`2.14`. Preserve the immutable result, retire exact-trace preload from future
Portal candidates, and do not infer the exact cause from aggregate metrics
alone.

## Current Queue After R033

No R023 artifact is eligible for Portal. The current champion bytes are the
I006 seq16 configuration at SHA-256
`e28f0aac456ee2206e3d9b51912829584681d0a0786e71330e519bdcfb1bbc4f`,
but that historical path has already been graded at `51.59` and must not be
mutated or presented as a new optimization.

R024 validation passed using all four selected trace controls and all four
fresh I006 quality controls. R025 then selected I007 seq19 locally and froze
Attempt 14 at:

```text
portal_submissions/attempt_14_vllm_i007_fp8_head_seq19.yml
SHA-256: 6db580dfb747cdbc3c23560d71cdd9645ea89316da51612d836cae009a37f025
image: sha256:d6d626e8f5f1a2db3bd6974557f55a72b73aff869bc2421e18a4feb67a3b7611
```

It changed I007 active concurrency from seq16 to seq19 and kept the existing
I005 warmup. Portal graded it `46.86`, with `19/120` passed SLO, TTFT
`4295/6586ms`, TBT `19ms`, accuracy drop `2`, penalty `1` and zero failures.
Seq19 transferred a TTFT-tail benefit but lost decode score. I006 seq16 at
`51.59` remains the conservative rollback.

R026b prepared one optional derivative:

```text
portal_submissions/attempt_15_vllm_i007_seq19_graph11.yml
SHA-256: 23f47a9c882083be03d5a8f746c0a3b365d14b4285d9f24c7906a945d7856608
delta: add graph size 11 only
```

Its local evidence is weak after warm-control confirmation, so it is an H200
calibration probe rather than a promoted champion. Graph6 and graph13 are not
included.

R027 prepared and Portal graded the seq18 interpolation candidate:

```text
portal_submissions/attempt_16_vllm_i007_seq18.yml
SHA-256: f0e5f647493aea5dde3e55d0e3cdf625226095db82b0fad741912084416082b3
delta from Attempt 14: seq19/graph19 -> seq18/graph18
```

Its local seq19/seq18/seq19 bracket passed all ten frozen probe gates. Portal
returned score `47.04`, `19/120` SLO passes, TTFT `4407/8799ms`, TBT `18ms`,
accuracy drop `0` and no failures. Relative to Attempt 14, decode improved by
`1ms` and score by `0.18`, but TTFT p95 regressed by `2213ms`. Reject seq18 as
a complete candidate and park graph11 Attempt 15 unless new H200 evidence
creates a specific reason to spend a calibration slot.

R028 tested the only remaining scalar budget uncertainty on exact public I006
seq16. In frozen `2048/4096/2048` order, fixed `4096` lost `1.43%` ERS and
TTFT score in all six waves despite better raw TPOT and six more requests under
the TTFT ceiling. The branch is rejected locally. No Attempt 17 image or
Compose was created; default/resolved `2048` remains in the I006 champion.

R029 also rejects dynamic Tree-Gated switching before GPU. Its robust oracle
gain is only `0.351%`, and depth-2 leave-one-wave-out ERS is negative. No
scheduler overlay, image or Compose was built; Attempt 17 remains unassigned.

R030 corrects the R028 aggregate metric explanation but creates no candidate.
Raw lookup hits fell with lookup queries; actual request cache attribution
changed only `-0.143%`. Request-level ERS and gates are unchanged, so R028 and
R029 remain closed and Attempt 17 remains unassigned.

R031/R031b then ran six exact-I006 lifecycle cells in both bracket orders.
Fixed `4096` reduced queue time but increased scheduled-to-first-token service,
lost `1.80%` mean ERS and lost five of six wave scores. The strict promotion
gate failed. No scheduler, image or Compose was built; Attempt 17 remains
unassigned.

R032 closed the final-chunk derivative offline. At `50%` net service recovery,
only `6/9` process pairs won, waves 1 and 5 lost, and the `<=500ms` cohort
regressed. No R032b, image or Compose was built; Attempt 17 remains unassigned.

R033 requalified DR03b quality under the official BTC penalty and then ran the
required same-image off/on/off bracket. Candidate ERS `0.612869` lost `0.257%`
to the control midpoint, W/T/L was `32/35/53`, and only three waves were
positive. The fresh quality run was skipped after performance failed. No image,
Compose, push or Attempt 17 artifact was created.

## Quy Tắc

- Không sửa file của một attempt sau khi đã nộp.
- Candidate mới phải dùng tên attempt mới.
- Ghi lại kết quả Portal vào `submission_plan.md` trước khi chuẩn bị attempt kế tiếp.
- Không đưa đường dẫn hoặc bind mount từ máy local vào compose.
- Custom code phải nằm trong public Docker image; Portal chỉ nhận compose.
- Không nộp lại Attempt 04/digest v1 hoặc Attempt 06/digest v3.
- Không nộp lại Attempt 07 như một artifact mới trước khi chọn rõ retry cùng
  digest hay chuyển sang image runtime nhỏ hơn.
- Attempt 05 đã chấm thành công. Historical Tree-Gated vLLM image đã được
  đóng gói và A/B đúng `3 CPU/8GB`, nhưng bị loại trước khi push: model chỉ tạo
  hai priority trùng hoàn toàn với hai đoạn arrival `0..39` và `40..119`, nên
  dispatch giống FCFS và proxy chỉ thêm overhead.
- Attempt 08 là direct official vLLM `seq8 + FP8 E4M3`; không sửa file sau khi
  checksum cuối đã ghi và không nộp lại artifact này.
- Attempt 09 là immutable `seq6 + FP8 E4M3` calibration đã chấm `49.98`.
  Không sửa hoặc nộp lại; giữ làm mốc H200 TBT `18ms`.
- Original Attempt 10/11 đã chấm `44.09/46.03`; giữ immutable để calibration.
  I006 seq16 SHA `e28f0a...` đạt `51.59`, TTFT `4783/6800ms`, TBT `16ms`,
  accuracy drop `0` và là complete-panel champion. Không trộn nó với original
  Attempt 11 seq20.
- Attempt 12 I007 seq16 top-k4 đã chấm `51.37`, accuracy drop `0`, TBT `15ms`.
  Đây là quality-safe watchlist dưới I006 `0.22`; không sửa hoặc nộp lại
  immutable file.
- Attempt 13 R022d is the completed single user-authorized exact-trace risk
  trial. It scored `49.45`; do not mutate or duplicate it, and do not describe
  successful grading as rules approval.
- Attempt 14 I007 seq19 is graded `46.86`. It added three SLO passes and
  improved TTFT tail versus seq16 I006, but TBT rose to `19ms`. Preserve it as
  TTFT attribution evidence and use R026b only to repair decode performance.
- Attempt 15 adds graph11 only. It is parked after Attempt 16 because the
  confirmed local graph11 effect is nearly zero and slightly worsens TPOT p99.
  Do not submit it without new target-specific causal evidence.
- Attempt 16 changes seq19/graph19 to seq18/graph18 only. It graded `47.04`:
  TBT improved to `18ms`, but TTFT p95 rose to `8799ms`. Preserve the artifact
  and close active-set interpolation; do not mutate or resubmit it.
- R028 fixed budget `4096` is locally rejected on exact I006. Do not package
  or submit it as Attempt 17, and do not reopen fixed budget `16384`.
- R029 dynamic `2048/4096` policy is closed with `0/6` opening gates. Do not
  train a larger policy or create an attempt from this branch.
- R030 is measurement-only. Do not package a cache-loss repair based on raw
  repeated prefix lookup counters; no Attempt 17 artifact was created.
- R031 is also measurement-only. Retain its lifecycle observer for diagnosis,
  but do not package fixed/dynamic `4096`; the bidirectional gate failed and
  Attempt 17 remains unassigned.
- R032 is offline-only and closes final-chunk isolation. Do not package its
  ideal phase-splice ceiling; no R032b or Attempt 17 artifact was created.
- R033 closes DR03b on same-image performance. Do not package the existing
  hardened cache or run its conditional quality panel; no Attempt 17 artifact
  was created.
- Không tạo candidate seq8-12 chỉ bằng cách tăng concurrency. Attempt 10 là
  ngoại lệ có chủ đích vì FP8 weights + BF16 Mamba state đã thay đổi decode
  path và vượt A/B/A/quality gate; nó chỉ dùng để calibration H200.
- Giữ model path `/model`, served name `Qwen3.5-2B`, host `0.0.0.0` và port `8000`.

Kế hoạch và decision tree nằm trong `submission_plan.md`.
