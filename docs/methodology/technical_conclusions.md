# Technical Conclusions

Date: 2026-07-07

## Scope

File này ghi lại các kết luận kỹ thuật đã rút ra từ baseline, Experiment 001 và các warmup rounds trên RTX 5080 local workspace.

Các kết luận ở đây là tín hiệu định hướng local, không phải kết luận cuối cho môi trường chấm H200 MiG của BTC.

## Current Best Local Baseline

Cấu hình tốt nhất đã đo được trên RTX 5080:

```env
GPU_MEMORY_UTILIZATION=0.72
MAX_NUM_SEQS=4
ENABLE_CHUNKED_PREFILL=true
MAX_MODEL_LEN=32768
```

Best run:

- Run: `local_workspaces/rtx5080/runs/20260707-204049`
- ERS: `0.500000`
- Success: `120/120`
- TTFT p50: `37711.23 ms`
- TPOT p50: `9.84 ms`
- TPOT p95: `19.72 ms`
- GPU memory max: `13059 MiB`

## Main Finding

Sweet spot hiện tại không nằm ở việc tăng `GPU_MEMORY_UTILIZATION` đơn thuần, mà nằm ở việc giới hạn scheduler concurrency đủ mạnh để decode không bị nghẽn, rồi cấp đủ GPU memory để runtime giữ TPOT ổn định.

Tăng KV cache giúp vLLM nhận nhiều request đang chạy hơn, nhưng khi batch quá lớn thì decode contention tăng mạnh, làm `TPOT` xấu đi và kéo ERS xuống.

Kết luận chính:

```text
More KV cache is not automatically better.
Controlled scheduling is more important than maximum concurrency.
```

## What Improved Score

Giới hạn `MAX_NUM_SEQS=4` cải thiện ERS rõ nhất trong Round 002; Round 003 nâng cấu hình local best lên `GPU_MEMORY_UTILIZATION=0.72`.

Lý do:

- Giảm số sequence decode đồng thời.
- Giảm cạnh tranh compute/memory trong decode.
- Đưa TPOT p50 xuống `10.03 ms`, thấp hơn floor chính thức `20 ms`.
- Giữ toàn bộ 120 request chạy thành công, không timeout/OOM/zero-token.
- Chấp nhận TTFT xấu hơn vì TTFT local đã vượt xa ceiling và không còn đóng góp điểm.

## What Did Not Work

Các cấu hình chỉ tăng hoặc giảm `GPU_MEMORY_UTILIZATION` với `MAX_NUM_SEQS=120` không tốt hơn:

| Config | ERS | Technical reading |
|---|---:|---|
| `0.75 / max_num_seqs=120` | `0.018118` | TTFT tốt hơn nhưng TPOT sập nặng. |
| `0.70 / max_num_seqs=120` | `0.015397` | Vẫn quá nhiều decode contention. |
| `0.60 / max_num_seqs=120` | `0.016209` | Ít concurrency hơn nhưng TPOT vẫn không đủ tốt. |
| `0.65 / max_num_seqs=120` rerun | `0.016112` | Không ổn định, TPOT xấu hơn run đầu. |
| `0.65 / max_num_seqs=16` | `0.057681` | Ổn định nhưng TPOT p50 vẫn cao hơn ceiling. |
| `0.65 / max_num_seqs=8` | `0.297791` | Tốt hơn 16 rõ rệt, nhưng TPOT vẫn kém 4. |
| `0.65 / max_num_seqs=4` | `0.500000` | ERS đã bão hòa ở mức local hiện tại, nhưng tie-break kém 0.72. |
| `0.68 / max_num_seqs=4` | `0.499667` | TPOT p95 vượt 20 ms nhẹ, làm mất chút điểm. |
| `0.72 / max_num_seqs=4` | `0.500000` | Best local run hiện tại theo tie-break TTFT/TPOT/throughput. |
| `0.72 / max_num_seqs=4 / max_batched_tokens=16384` | `0.500000` | TPOT tốt hơn nhưng TTFT và VRAM xấu hơn; không promote. |

## Bottleneck Interpretation

TTFT vẫn là bottleneck lớn nhất nếu nhìn theo production objective đầy đủ.

Tất cả các run local hiện tại đều có TTFT cao hơn rất nhiều so với ceiling chính thức `1500 ms`. Vì vậy ERS local chủ yếu được cứu bởi những request có TPOT còn điểm. Round 002 cho thấy khi TTFT đã bị 0 điểm, giảm TPOT xuống dưới floor có thể tăng ERS rất mạnh. Round 003 cho thấy local ERS đã chạm `0.5`, tức là TPOT gần như không còn là đòn tăng điểm chính nếu TTFT vẫn bằng 0 điểm.

Điều này cho thấy:

- Prefill workload quá nặng do prompt rất dài.
- Queue pressure lớn vì trace bắn 120 request với arrival dày.
- KV cache và scheduler quyết định hiệu năng tổng thể.
- Decode-only optimization chưa đủ nếu TTFT vẫn quá cao.

## Practical Rule For Next Experiments

Không tối ưu theo một metric đơn lẻ.

Nếu tăng concurrency:

- TTFT có thể giảm.
- Throughput có thể tăng.
- Nhưng TPOT có thể xấu đi đủ nhiều để ERS giảm.

Nếu giảm concurrency:

- TPOT có thể tốt hơn.
- Nhưng waiting time và TTFT có thể tăng.

Mục tiêu là tìm điểm cân bằng giữa queueing và decode contention.

## Current Hypothesis

Hướng có khả năng cải thiện tiếp:

1. Kiểm tra warmup/JIT vì vLLM vẫn log compile/JIT trong inference path.
2. Kiểm tra prefix cache behavior vì trace có prompt dài và nhiều message.
3. Giữ `MAX_NUM_SEQS=8` làm fallback nếu các vòng sau cải thiện TTFT/prefill và cần tăng concurrency.
4. Chỉ quay lại memory tuning nếu có OOM, instability hoặc đổi runtime/framework.

## Round 002 Finding

Round 002 đã chạy đủ các candidate `MAX_NUM_SEQS=4, 8, 12, 16, 20, 24, 32`.

Kết quả quan trọng:

- `MAX_NUM_SEQS=4`: ERS `0.499987`, TPOT p50 `13.91 ms`, TTFT p50 `49884.60 ms`.
- `MAX_NUM_SEQS=8`: ERS `0.297791`, TPOT p50 `25.19 ms`, TTFT p50 `42458.59 ms`.
- `MAX_NUM_SEQS=16`: ERS `0.057681`, TPOT p50 `50.34 ms`, TTFT p50 `38725.75 ms`.

Diễn giải: giảm concurrency làm TTFT xấu hơn nhưng TPOT tốt hơn rất nhiều. Với scoring hiện tại, TPOT thắng vì TTFT local đang ở ngoài vùng có điểm.

## Round 003 Finding

Round 003 đã chạy đủ các candidate `GPU_MEMORY_UTILIZATION=0.58, 0.62, 0.65, 0.68, 0.72` với `MAX_NUM_SEQS=4`.

Kết quả quan trọng:

- `0.72`: ERS `0.500000`, TPOT p50 `10.03 ms`, TTFT p50 `38655.35 ms`, GPU max `13359 MiB`.
- `0.65`: ERS `0.500000`, TPOT p50 `13.45 ms`, TTFT p50 `41363.54 ms`, GPU max `12412 MiB`.
- `0.68`: ERS `0.499667`, TPOT p95 `20.06 ms`, mất nhẹ điểm TPOT.

Diễn giải: `0.72` là best local config hiện tại theo tie-break, nhưng mức tăng điểm đã bão hòa. Việc tiếp tục tối ưu TPOT không đủ; round sau phải nhắm TTFT/prefill/queueing.

## Round 004 Finding

Round 004 đã chạy đủ các candidate `MAX_NUM_BATCHED_TOKENS=default, 8192, 16384, 24576, 32768, 49152` với `GPU_MEMORY_UTILIZATION=0.72` và `MAX_NUM_SEQS=4`.

Kết quả quan trọng:

- `16384`: ERS `0.500000`, TPOT p50 `8.24 ms`, TTFT p50 `40307.05 ms`, GPU max `15283 MiB`.
- `default`: ERS `0.499731`, TPOT p50 `9.82 ms`, TTFT p50 `37608.58 ms`, GPU max `14388 MiB`.
- `32768` và `49152`: TTFT p50 tăng rất mạnh lên `64557.05 ms` và `78274.93 ms`.

Diễn giải: `max_num_batched_tokens` có thể cải thiện TPOT, nhưng không xử lý bottleneck TTFT. `16384` là watchlist nếu cần TPOT đẹp hơn, còn global local baseline vẫn là Round 003 vì cân bằng TTFT/VRAM tốt hơn.

## Round 005 Finding

Round 005 đã chạy chunked prefill candidates:

- `explicit_on`: ERS `0.500000`, TTFT p50 `37711.23 ms`, TPOT p50 `9.84 ms`, GPU max `13059 MiB`.
- `default`: ERS `0.499057`, TTFT p50 `37556.66 ms`, TPOT p99 `20.65 ms`.
- `no_chunked`: không ready trong 10 phút.
- `max_num_partial_prefills > 1`: không ready trong 10 phút.

Diễn giải: chunked prefill phải được bật rõ ràng. Tắt chunked prefill hoặc bật concurrent partial-prefill không phù hợp với local resource limit hiện tại. Global local baseline được nâng lên Round 005 `explicit_on`.

## Round 008/009 Brain 1 Finding

Round 008 tạo dataset shadow RAM/VRAM-only cân bằng từ 7080 request-run rows:

- RAM: `3540`
- VRAM: `3540`
- Không dùng SSD/offload làm target vì môi trường BTC không đảm bảo SSD path hữu dụng/đồng đều.

Round 009 train shadow ranker bằng pure-NumPy boosted stumps vì local chưa có `xgboost`, `sklearn`, `pandas`.

Kết quả production-safe chính:

- Target: `ram_vram_target`
- Split: `request_id`
- Feature set: `trace_pressure`
- ROC AUC: `0.973690`
- Balanced accuracy: `0.932974`
- F1: `0.931900`
- Top features: `timestamp_ms`, `vram_usage_proxy`, `surviv_index`

Kết quả risk ranking:

- `score_proxy_bad` với `trace_pressure`: ROC AUC `0.871762`
- `tpot_over_ceiling` với `trace_pressure`: ROC AUC `0.905334`

Diễn giải: Brain 1 đã có thể học policy shadow RAM/VRAM và risk TPOT/score từ các biến request + pressure. Nhưng đây chưa phải tối ưu serving thật. Nó chỉ là bộ lọc/ranker để Round 010 mô phỏng admission hoặc placement trước khi đụng vào vLLM.

Lưu ý quan trọng:

- `trace_pressure_config` đạt điểm cao hơn ở vài target, nhưng có thể dựa vào field cấu hình cố định như `dtype` và `max_num_seqs`. Dùng nó cho phân tích, không lấy làm primary production-safe model khi submission chỉ chạy một cấu hình cố định.
- Không train được classifier TTFT-pass vì toàn bộ local rows đều vượt ceiling TTFT `1500 ms`.

Round 009b đã sweep hyperparameter để tránh dùng model default một cách chủ quan:

- Broad sweep: `48` config
- Deep shortlist: `60` eval
- Sweet spot: `n60.lr0.200.thr8.leaf24`
- Request ROC AUC / balanced accuracy / F1: `0.972375 / 0.932974 / 0.931900`
- Run ROC AUC / balanced accuracy / F1: `0.946675 / 0.909237 / 0.900405`
- Risk AUC mean: `0.854707`
- Top features: `message_count`, `vram_usage_proxy`, `surviv_index`, `entropy_score`

Diễn giải: Round 009 default đã gần tối ưu, nhưng `009b` chọn model nhỏ hơn và regularized hơn (`60` estimators, `8` thresholds, `min_leaf=24`) để giảm rủi ro overfit khi đem sang Round 010 simulation. Nếu cần biến thể mạnh hơn, fallback là `n60.lr0.200.thr16.leaf24`.

## Round 010 Brain 1 Simulation Finding

Round 010 dùng model sweet spot từ Round 009b để replay offline các policy admission/placement.

Kết quả chính:

- Best baseline: `safe_seq4`, ERS `0.500000`, TTFT p50/p95 `18810.979500 / 31400.715950 ms`
- Best policy quality: threshold `p_vram >= 0.15`
- Policy balanced accuracy: `0.976190`
- VRAM precision/recall/F1: `0.975000 / 1.000000 / 0.987342`
- Best hybrid proxy: top `40%` p_vram dùng `aggressive_seq12`, còn lại dùng `fp8_full_score`
- Hybrid ERS: `0.500000`
- Hybrid TTFT p50/p95: `8016.931000 / 28315.949750 ms`
- Best observed per-request oracle ERS: `0.500000`
- Min-TTFT oracle ERS: `0.450347`

Diễn giải: Brain 1 policy có tín hiệu thật để chọn request ưu tiên, và proxy tốt nhất giảm TTFT p50 từ `18.8s` xuống `8.0s` mà không mất ERS. Nhưng TTFT vẫn xa ceiling `1500 ms`, nên điểm không tăng. Đây là negative/guardrail result: không đưa Brain 1 admission vào serving/submission lúc này.

Kết luận thực chiến: Brain 1 nên giữ làm analysis/ranker. Muốn tăng điểm phải xử lý TTFT/prefill bằng runtime/scheduler thật, không chỉ policy replay.

## Round 012 Warmup/JIT Finding

Round 012 kiểm tra warmup/JIT/startup và mức VRAM usable cao hơn trên RTX 5080 local với giới hạn Docker `3 CPU`, `8GB RAM`.

Kết quả môi trường quan trọng:

- `GPU_MEMORY_UTILIZATION=0.95` không chạy được trên phiên desktop hiện tại.
- vLLM báo free startup memory `12.91/15.44 GiB`, trong khi `0.95` cần `14.67 GiB`.
- `nvidia-smi` cho thấy desktop/app đang giữ khoảng `2.1GB` VRAM trước khi serve.
- `0.83` là experimental near-max usable VRAM, nhưng flaky vì một candidate failed khi free chỉ còn `12.76 GiB` còn vLLM cần `12.81 GiB`.
- `0.82` là stable high-VRAM fallback cho phiên desktop này.

Kết quả benchmark:

- Cold `0.83`: ERS `0.500000`, TTFT p50/p95 `17673.4095 / 30627.24865 ms`, TPOT p50/p95 `8.4574325 / 9.740635 ms`.
- Torch warmup `10s` ở `0.83`: ERS `0.500000`, TTFT p50/p95 `17421.6925 / 31319.2015 ms`, TPOT p50/p95 `8.71467 / 9.482708 ms`.
- Cold `0.82`: ERS `0.500000`, TTFT p50/p95 `18539.6135 / 32264.74145 ms`, TPOT p50/p95 `8.6779925 / 10.066987 ms`.
- Endpoint short warmup `0.82`: ERS `0.5006135105048746`, TTFT p50/p95 `17849.3315 / 31575.4577 ms`, TPOT p50/p95 `8.6846975 / 10.096045 ms`.

Endpoint warmup cost:

- 3 synthetic streaming requests.
- TTFT: `938.286 ms`, `23.262 ms`, `22.321 ms`.

Diễn giải:

- Endpoint warmup có tác dụng thật: so với cold `0.82`, ERS tăng `+0.0006135`, TTFT p50/p95 giảm khoảng `690 ms`.
- Pure Torch occupancy warmup yếu hơn endpoint warmup trong round này. `10s` có cải thiện TTFT p50 nhưng trade-off tail/throughput.
- Warmup chỉ che được một phần first-endpoint/JIT cost; bottleneck chính vẫn là queue/prefill trên full trace.

Kết luận thực chiến:

- Giữ endpoint warmup làm kỹ thuật watchlist.
- Không submit chỉ vì Round 012: score proxy tốt nhất chỉ `50.061351`, còn xa target `95`.
- Round tiếp theo nên là subround endpoint-shape warmup: prompt synthetic medium/long và 4-way concurrent warmup để cover nhiều JIT/prefill shape hơn.

### Round 012a High VRAM Subround

Sau khi VRAM free tăng lên khoảng `14594 MiB`, Round 012a sweep lại `GPU_MEMORY_UTILIZATION` từ `0.82` tới `0.90`.

Kết quả capacity:

- `0.82`: KV cache `6.13 GiB`, `883850` tokens, max 32K concurrency `26.97x`.
- `0.90`: KV cache `7.36 GiB`, `1062745` tokens, max 32K concurrency `32.43x`.
- Capacity tăng `+178895` KV tokens, khoảng `+20.24%`.

Kết quả score/latency:

- `0.82`: ERS `0.500000`, TTFT p50/p95 `16462.7585 / 28407.40815 ms`, TPOT p50/p95 `8.09667 / 9.375602 ms`.
- `0.90`: ERS `0.500000`, TTFT p50/p95 `16504.6455 / 28455.75355 ms`, TPOT p50/p95 `8.0987875 / 9.40156925 ms`.

Diễn giải:

- VRAM cao hơn có hiệu quả thật về KV capacity.
- Nhưng với `MAX_NUM_SEQS=4`, workload hiện tại không dùng hết extra capacity, nên score/latency không tăng.
- Nếu muốn tận dụng H200 MiG `18GB`, phải ghép VRAM cao với scheduler/concurrency/prefill strategy. Chỉ tăng `gpu_memory_utilization` riêng lẻ không đủ.

## Ready 008a Engine-Native Scheduler Finding

Ready 008a đo trực tiếp scheduler của SGLang `0.5.14` trên đúng candidate
SGLang hiện tại, không thay đổi queue order hay budget.

Kết quả parity:

- Native control ERS `0.512927`.
- Instrumented ERS `0.511176`.
- Delta `-0.001751`, success `120/120`, TPOT p95 `17.918 ms`.
- Telemetry đủ `120 queued / 120 first-token / 120 finished`, không lỗi JSONL.

Kết quả workflow:

- 237 prefill batch, 2800 decode batch, không có mixed batch.
- Prefill batch size trung bình `1.447`; decode `8.571`.
- Prefix match ratio ở first prefill trung bình `0.7767`.
- Prefill scheduling path trung bình/p95 `73.910/90.429 ms`; riêng LPM
  calculation chỉ `2.134/4.334 ms`.
- 343 unfinished chunk commit, 51 eviction call, 114461 full-KV token bị
  evict, nhưng không có Mamba-state eviction và không có request retraction.

Diễn giải:

- Bottleneck hiện tại là admission theo wave kết hợp chunk/cache bookkeeping,
  không đơn giản là LPM sort chậm.
- Prefix reuse mạnh nên vẫn phải giữ LPM làm default.
- Global FCFS/LOF, global mixed chunking và HTTP proxy đều đã có negative
  evidence.
- Hướng tiếp theo là conditional prefix rescue trong engine: chỉ cho một
  request có cache hit cao chen vào decode wave nếu dự đoán còn khả năng đạt
  TTFT `<=1500 ms`, đồng thời giữ TPOT p95 `<=20 ms`.

## Ready 006k Cache-Local Learned Scheduler Finding

Ready 006k kiểm tra trực tiếp Tree-Gated v2 và heuristic bốn biến trong boundary
an toàn nhất: chỉ so request có cùng non-root native cache node sau LPM prefix
matching.

Kết quả offline:

- Heuristic thắng actionable AUC: `0.916530` so với Tree `0.900176`.
- Hai policy đồng ý phần lớn nhưng không hoàn toàn trên replay tổng quát.

Kết quả live:

- Có `263` exact-node cohort observations và `60` node tags khác nhau.
- Tree would-reorder `0`; heuristic would-reorder `0`.
- Không có policy error, queue mutation, active rescue hoặc thay đổi native order.
- Evaluator p99 `333.063us`, cao hơn budget `250us`.
- ERS control/shadow `0.524547 / 0.518136`, delta `-0.006411`.
- Shadow có TTFT p50/p95, TPOT và throughput tốt hơn nhẹ, nhưng số request
  TTFT<=1500 giảm `25 -> 20`; đây là minh chứng distribution quanh scoring
  ceiling quan trọng hơn percentile đơn lẻ.

Diễn giải:

- Exact cache-node là locality boundary hợp lệ nhưng quá hẹp cho bộ feature
  hiện tại.
- Native position chi phối model/heuristic, nên cả hai chỉ học lại đầu hàng LPM.
- AUC cao không có giá trị điều khiển nếu policy không tạo action khác baseline.
- Không nên tốn thêm benchmark chỉ để giảm overhead của một action set rỗng.

Kết luận thực chiến: giữ Tree/C++ và hook làm research infrastructure, không
đưa vào Compose/Portal. Hướng learned scheduler tiếp theo chỉ được mở khi
offline held-out replay chứng minh nonzero action support với feature mới và
cohort rộng hơn nhưng vẫn giữ cache locality.

## Ready 006l Parent-Ancestry Replay Finding

006l kiểm tra điều kiện mở lại learned scheduler hoàn toàn offline.

Topology:

- Cache graph có `146` node quan sát được, không có parent conflict.
- Direct parent có `276` same-key groups nhưng `0` distinct current node.
- Ancestor 2/3/4/5 hops có `248/217/49/11` groups nhưng vẫn `0` distinct node.
- Normalize theo tree depth 3/4/5 có `165/164/160` groups, vẫn `0` distinct node.

Vì vậy current `last_node` ancestry không tạo được cohort rộng hơn exact-node
trong scheduler decision của trace này.

Khi bỏ hoàn toàn native position trong exact-node cohort:

- `service_only` đổi `126/216` validation selections.
- Native giữ `6/6` positive cohort heads, candidate giữ `0/6`.
- TTFT component giảm `0.002374`.

Diễn giải: native position trong LPM không chỉ là feature dễ học; nó đang nén
thông tin arrival/cache progression cần thiết để bảo vệ nhóm TTFT-positive hiếm.
Static suffix-shortest/deadline heuristic phá tín hiệu này.

Kết luận thực chiến: đóng nhánh static cache-cohort reranking. Hướng tiếp theo
nên tối ưu budget prefill/decode/admission hoặc thu temporal causal state mới;
không tiếp tục đổi ranker trên cùng snapshot.

## Score 100 R009-R010 Warmup And Packaging Finding

Warmup phải cover đúng execution shape, không chỉ tạo GPU occupancy. R009 dùng
một synthetic chat prompt `2710` token để vượt budget prefill `2048`, sau đó
20 synthetic short request chạy đồng thời để cover wave20/decode. Kế hoạch này
đưa toàn bộ năm JIT kernel quan sát được ra khỏi timed traffic.

Kết quả runtime-only R009:

```text
ERS candidate/control midpoint: 0.617657 / 0.581198
TTFT p50/p95:                   1020/2454ms
TPOT p95/p99:                   17.364/19.688ms
timed JIT:                      0
```

R010 chứng minh có thể đưa cùng warmup vào trước readiness mà vẫn chỉ load một
model. Wrapper giữ external module BTC, chạy native vLLM backend ở port 8001,
warmup xong mới bind transparent relay port 8000.

Kết quả packaged I006:

```text
ERS candidate/control midpoint: 0.619885 / 0.589324 (+5.19%)
TTFT p50/p95:                   1008/2414ms
TPOT p95/p99:                   17.117/19.269ms
TTFT <=1.5s:                    93/120
public readiness:               102s
timed JIT:                      0
```

Một A/B/A khác tắt warmup để cô lập relay: ERS relay `0.594723` so với direct
midpoint `0.592591`. Đây chỉ là bằng chứng relay không tạo regression đo được,
không phải relay tự tăng tốc model.

Kết luận thực chiến:

- promote I006 làm local packaged champion;
- warmup phải synthetic/trace-independent và hoàn tất trước public readiness;
- không dùng request ID, precomputed response hoặc exact-trace state trong
  I006; R022b chỉ được nghiên cứu local dưới gray-area submission hold;
- không load model thứ hai và không thêm request-aware scheduling vào relay;
- giữ I004/Attempt 10 làm clean no-warmup control;
- dừng local integration tại đây cho đến khi H200 cho biết warmup Hopper có lợi
  hay chỉ là đặc tính JIT của RTX 5080.

## H200 Attempt 09 Decode Calibration

Attempt 09 seq6+FP8 KV trả về:

```text
score:          49.98
TTFT p50/p95:  21496/24498ms
TBT median:     18ms
passed SLO:     3/120
accuracy drop: 2
penalty:        1
```

Local projection là `17.94ms`, gần như đúng tuyệt đối với H200 `18ms`. Vì vậy
decode/TBT scaling từ cùng vLLM path có giá trị để chọn boundary. Tuy nhiên
TTFT local không chuyển thành score đủ tốt: so với Attempt 05, TTFT p50 chỉ
giảm khoảng `10.5%`, trong khi TBT tăng `13 -> 18ms` và score giảm
`50.13 -> 49.98`.

Kết luận: không tiếp tục tối ưu chỉ bằng seq4/6/8. Attempt 10 vẫn cần đo vì nó
thay đổi weight bandwidth, Mamba-state block và graph shape. Rough projection
từ I004 local TBT median `11.60ms` theo hệ số seq6 khoảng `1.83x` là gần
`21.3ms`; đây là risk estimate, không phải dự đoán chắc chắn vì online FP8
weights có thể scale khác trên Hopper.

## H200 Attempts 10/11 And R011 Boundary

Attempt 10/11 đã trả lời phần còn mở ở trên:

```text
                         Attempt 10 I004   Attempt 11 I006
score                    44.09             46.03
TTFT p50/p95             4522/5422ms       4306/5334ms
TBT median               21ms              20ms
passed SLO               22/120            20/120
accuracy drop/penalty    0/1               2/1
failed/warmup            0/0               0/0
```

I004 chuyển TTFT lên H200 tốt hơn hẳn seq4/6, nhưng decode pressure làm TBT
chạm hoặc vượt floor. I006 warmup/package có lợi thật trên H200: score tăng
`1.94`, TTFT giảm và TBT `21 -> 20ms`; `warmup_count=0` cũng xác nhận warmup
nội bộ không bị tính thành traffic. Tuy nhiên cả hai vẫn thua Attempt 05
`50.13`.

R011 thử giảm active set với đúng graph tương ứng. Seq18 là local Pareto
candidate (`ERS 0.593090`, TPOT p99 `18.457ms`, `814.132 tok/s`) nhưng không
đạt calibrated TPOT p50 target. Seq17 gần như không thêm ERS và chỉ còn ít
throughput margin trên ngưỡng drain wave `800 tok/s`.

Kết luận thực chiến:

- đóng nhánh giảm concurrency thuần;
- giữ seq18/graph18 làm decode guard có điều kiện, không làm champion;
- profile riêng 18 Mamba/GDN và sáu full-attention layer trước khi viết kernel;
- ưu tiên fused norm/quant, Mamba decode, Mamba prefill và hybrid state;
- chỉ mở lại singleflight/two-policy sau khi HybridRadixState tạo action mới;
- roadmap thực thi tiếp theo là R012-R025 trong
  `docs/planning/score_100_research_roadmap.md`.

## R012-R017 Kernel Conclusions

R012 cho thấy decode bottleneck lớn nhất không nằm ở full attention mà nằm ở
tied BF16 LM head:

```text
pure decode step batch20: 3.9795ms
tied LM head:             1.1381ms, 28.6%
GDN / norm / attention:   8.4% / 6.4% / 1.2%
```

R013 dùng một bản E4M3 chỉ cho output projection, giữ embedding BF16, rồi sửa
chính xác top-4 logits bằng weight BF16. Hidden-state replay khôi phục top-1
`500/500`; full trace tăng ERS `0.514707 -> 0.532424` theo stable controls;
GPQA giảm `35 -> 33/100` nhưng penalty proxy vẫn `1`. Đây là kernel candidate
duy nhất được giữ, chưa phải integrated champion vì tốn thêm `485MiB` và giảm
KV capacity `6.63%`.

R014-R017 tạo các boundary âm tính quan trọng:

- native norm/quant fusion match `0` graph pattern;
- launch geometry GDN chỉ đạt `1.0015x`;
- FP8/INT8 recurrent state giảm gần nửa memory nhưng chậm hơn BF16 `4-6%` và
  tích lũy sai số sau 200 step;
- FLA chunk32 chậm hơn chunk64, chunk128 chưa được pipeline hỗ trợ;
- local CuteDSL package lỗi, local FlashInfer GDN binary chỉ hỗ trợ SM90;
  H200 đã được `auto` chọn đúng FlashInfer nên không ép backend.

Kết luận thực chiến:

- giữ R013 top-k4 để package A/B/A riêng;
- không đưa R014-R017 vào submission;
- giữ BF16 Mamba state, FLA chunk64 local và backend `auto`;
- R018 phải kiểm tra upper bound HybridRadixState trước khi viết engine vì
  cache hit `96.8%` vẫn recompute `219.5` token/branch.

## R018 HybridRadixState Upper Bound

State đúng của Qwen3.5-2B không chỉ là KV. Mỗi một trong 18 layer tuyến tính
cần giữ cả convolution state `[3,6144]` và recurrent state `[16,128,128]`,
tổng `561152` byte BF16/layer và `10100736` byte cho toàn model.

Hai endpoint trie chính xác cho thấy:

```text
native block560 measured compute: 456542 tokens
ideal block16 projected compute:  429646 tokens
maximum reduction:                26896 tokens / 5.89%
aggregate GPU-work upper bound:   611.1ms
largest-wave upper bound:         180.0ms / 4.18% Portal TTFT p50
```

Endpoint-only snapshot fit trong khoảng `32.4%` cache pool. Ngược lại, muốn
giữ state tại mọi radix boundary cần `108.91%` pool ngay ở block560 và
`2911.53%` ở block16. Hai phía trade-off không giao nhau: thiết kế fit VRAM
thì không giảm đủ compute, thiết kế giữ đủ boundary thì OOM.

Kết luận thực chiến:

- reject R018 trước khi viết continuation/allocator overlay;
- không mở R019 singleflight vì không có primitive state-sharing mới;
- không mở R020 two-policy vì thiếu causal state feature đã đặt làm dependency;
- giữ native BF16 align-mode endpoint;
- chuyển sang package A/B/A R013 trong R021a.

## R021 Packaged Tied-Head And Seq16 Balance

R021a loại bỏ confound của R013 bằng cách dùng đúng một image I006-derived cho
control/candidate/control; chỉ biến môi trường feature flag thay đổi. Kết quả:

```text
                              control midpoint   top-k4
ERS                           0.623107           0.626978
TTFT p50/p95                  988.684/2974.950    962.477/2866.026ms
TPOT p95                      15.064              14.492ms
throughput                    813.584             819.557 tok/s
peak VRAM                     14992               15087MiB
GPQA repeated                 34/100 x2           35/100 x2
```

Candidate thắng TPOT paired `120/120` so với cả hai control, thắng TTFT
`111/120` và `116/120`, không có setup/native/eager fallback, và penalty proxy
vẫn là `1`. Kết luận: output-only E4M3 tied head cùng exact BF16 top-k4 repair
là kernel đầu tiên vượt đủ package, performance và quality gate; promote thành
local I007.

R021b kiểm tra active set sau khi decode đã nhanh hơn. Seq17 chỉ hơn seq16
repeat `0.000747` ERS nhưng TPOT p99 xấu hơn `0.863ms`. Trong khi đó hai phiên
seq16 khác nhau lệch `0.006350` ERS, cho thấy nhiễu phiên lớn hơn nhiều so với
chênh lệch cấu hình. Seq15 cũng không đổi TPOT margin thành ERS.

Kết luận thực chiến:

- giữ `max-num-seqs=16`, không đẩy lên 17 chỉ vì một chênh lệch dưới nhiễu;
- Attempt 12 đã chấm đúng I007 seq16 tại digest `d6d626e8...b7611`: score
  `51.37`, ERC `0.133333`, TTFT p50/p95 `4148/7130ms`, TBT `15ms`, accuracy
  drop `0`, penalty `1`, không có failure;
- I006 seq16 native head tại Compose SHA `e28f0a...` đạt `51.59`, cao hơn
  I007 `0.22` điểm (`0.43%`). Full panel I006 vẫn thiếu, nên chưa biết chênh
  lệch đến từ TTFT hay TBT, nhưng tied-head không còn là H200 score champion;
- so với Attempt 11 seq20, score tăng `5.34`, TBT giảm `20 -> 15ms` và TTFT
  p50 giảm `4306 -> 4148ms`, nhưng TTFT p95 tăng `5334 -> 7130ms`;
- I007 local score proxy `62.6978` chỉ dùng so sánh RTX 5080, không phải dự báo
  leaderboard H200;
- output-head đã qua hidden quality nhưng chỉ giữ ở watchlist; H200 target quay
  về I006 seq16 native head;
- bước tiếp theo là một A/B/A head-off seq16/17/16 duy nhất trên đúng I006.
  Chỉ mở H200 seq17 nếu lợi ích vượt nhiễu, tăng fast-TTFT và không làm TPOT
  p99 xấu quá `1ms` hoặc vượt `20ms`.

R021c đã trả lời âm tính cho phép thử đó:

```text
seq16 midpoint / seq17 ERS: 0.607739 / 0.604517 (-0.530%)
TTFT <=1.5s:                78 / 78
TTFT score mean:            0.215478 / 0.209033
TPOT score mean:            1.0 / 1.0
TPOT p99:                   17.778 / 18.651ms
```

TTFT p95 seq17 giảm nhẹ `17.4ms`, nhưng p50 xấu hơn và toàn bộ score shape
xấu hơn. Vì TPOT score đã bão hòa, ERS giảm hoàn toàn do TTFT. Kết luận: giữ
I006 seq16, không nộp seq17 và không sweep tiếp seq18 nếu chỉ tăng concurrency.
Muốn mở R022 phải có full panel H200 `51.59` và một cơ chế mới tác động đúng
prefill/admission hoặc decode residual.

## R022a FP8 Linear Backend Screen

R012 cho thấy GEMM chiếm `68.7%` cold-prefill GPU time, nhưng resolver của
I006 đã chọn `CutlassFP8ScaledMMLinearKernel` ở chế độ `auto`. Ép
`--linear-backend=cutlass` cũng chọn đúng kernel đó và chỉ khác startup
`102 -> 103s`.

Các đường `flashinfer_cutlass`, `flashinfer_cudnn`,
`flashinfer_trtllm` và `triton` không implement được ít nhất một FP8 layer;
chúng fail startup thay vì tạo candidate chạy được. Vì vậy không có backend
khác biệt để bench full trace. Giữ `auto`, không đổi TTFT lấy TPOT, và khóa
mọi candidate tương lai ở TPOT p99 `<=18ms`.

## R022b Exact-Trace State Prime

R022b proves three distinct layers that must not be conflated:

1. raw prompt staging is not KV/state reuse;
2. priming 20 maximal conversation leaves reuses most history but not every
   intermediate request's terminal generation state;
3. priming all 120 exact prompts raises timed hit rate `96.24% -> 98.61%` and
   ERS `0.937695 -> 0.957285` under the same seq20 engine.

The all120 candidate reaches TTFT p50 `84.9ms`, p95 `398.3ms`, p99 `445.5ms`,
TPOT p99 `7.926ms`, success `120/120`, and Ready in `113s`. Its prime takes
`13.601s`; this is real model prefill moved before readiness, not
computation-free loading. GPQA remains deterministic with local drop `0.02`
and penalty proxy `1`.

Seq20 is correct only after the exact state is resident: it raises ERS
`8.524%` over the seq16 midpoint and removes every TTFT ceiling miss while
keeping large TPOT margin. xxhash is not the remaining solution; its ERS is
`0.215%` below SHA256 midpoint. The residual correlates with prompt length
before cache reuse and points to tokenization/request preprocessing.

Kết luận thực chiến:

- retain exact120+seq20 only as the gray-area local mechanism champion;
- keep SHA256 and reject xxhash;
- isolate original-tokenizer memoization next, without changing tokens or
  model output;
- do not create a Portal artifact until the compliance hold is resolved.

## R022c Original-Tokenizer Memoization

Once exact hybrid state is resident, the remaining TTFT grows with prompt
length because vLLM still renders and tokenizes every long chat before looking
up cached blocks. R022c caches the native renderer result during exact120
prime and returns an independent copy on timed hit. Misses stay on the native
path; token IDs, template semantics and model execution are unchanged.

Same-image off/on/off evidence:

```text
control midpoint / candidate ERS: 0.942855 / 0.999870 (+6.047%)
TTFT p50/p95/p99:                 42.9/80.4/96.6ms
TTFT <=100ms / max:               119/120 / 122.095ms
TPOT p99 / max:                   9.341 / 9.427ms
timed renderer hits:              120/120
startup:                          118s
```

Key generation and deep-copy work total only `82.655ms` and `262.476ms` over
all 120 timed requests. GPQA off/on is exactly `34/100` four times with zero
prediction flips. Therefore request preprocessing, not hash algorithm or GPU
decode, was the final local gap after exact state residency.

Final jitter closure tested two shortcuts. Disabling access logs regressed ERS
to `0.999816` versus deep-copy midpoint `0.999975`. Returning shared cached
objects reached the same ERS `1.000000` as both deep-copy controls but creates
cross-request mutation risk. Both are rejected.

The retained deep-copy configuration reached exact ERS `1.000000` three times
across the two follow-up brackets. Its latest run has TTFT
p50/p95/p99/max `43.5/70.3/81.3/84.0ms`, TPOT p99/max `9.000/9.071ms`, and
all `120/120` requests receive both full score components. Keep SHA256,
default access logging and independent deep copies.

Kết luận thực chiến: R022c reaches repeatable local Score 100, but it is more
benchmark-specific than R022b and remains strictly local. Do not confuse a
passed quality gate with a passed compliance gate.

## R022d Self-Contained Package Conclusion

Decision 058 opened exactly one user-accepted risk trial. R022d demonstrates
that R022c's result survives packaging rather than depending on a host trace
mount or a development-only process:

```text
Ready:                    118s
ERS / success:            1.000000 / 120/120
TTFT p50/p95/p99/max:     42.4/78.6/89.9/93.2ms
TPOT p95/p99/max:         8.776/9.160/9.227ms
packaged GPQA:            34/100 twice, zero reference flips
public linux/amd64 image: anonymous manifest pass
```

The image embeds only the public trace and stores no generated output. The
technical conclusion is narrow: pre-readiness exact hybrid-state prime plus
native-render memoization is reproducible in a self-contained vLLM package.
It does not establish that BTC permits benchmark-specific future-request
preprocessing.

Attempt 13 later graded at score `49.45`, TTFT p50/p95 `4066/5367ms`, TBT
`21ms`, `20/120` passed SLO, zero failures, accuracy drop `0` and penalty `1`.
Compared with I006 seq16 at `51.59`, it buys `717/1433ms` TTFT and four SLO
passes but loses `5ms` TBT and `2.14` score. The local exact-cache fast path
did not transfer materially. Close exact-trace derivatives, retain I006 seq16
as Portal champion, and treat the graded run as execution evidence rather than
explicit BTC rule approval.

## R022e/R022f Native MTP Conclusion

Qwen3.5's own MTP path was the lowest-risk speculative candidate because it
uses the BTC-fixed checkpoint rather than replacing it:

```text
model config: text_config.mtp_num_hidden_layers = 1
checkpoint: mtp.* weights exist
exact I006 image: qwen3_5_mtp and spec-tokens support exists in packaged source
```

Capability passed. At local fraction `0.78`, the exact image reached Ready in
`127s`, resolved `Qwen3_5MTP`, shared the target embedding/lm_head, loaded no
second full model and captured bounded piecewise graphs. The first three
memory fractions were blocked before weight load by desktop VRAM availability,
not by MTP or the fixed model hash.

The seq16 native/MTP/native bracket then rejected the technique:

```text
native midpoint / MTP ERS:       0.552309 / 0.466266 (-15.58%)
native midpoint / MTP TTFT p50:  1696.6 / 3336.9ms (+96.68%)
native midpoint / MTP TTFT p95:  4265.2 / 6210.6ms (+45.61%)
native midpoint / MTP TPOT p99:  22.147 / 25.611ms (+15.64%)
native midpoint / MTP throughput:768.512 / 700.116 tok/s (-8.90%)
MTP drafts / accepted:           14,275 / 9,656 (67.643%)
```

The mechanism is real, but its acceptance does not repay draft/verify cost and
the FlashInfer graph downgrade from full-and-piecewise to piecewise. Reject
MTP-1, do not run more draft tokens, skip GPQA because performance failed
first, and close dependent concurrency/top-k4 work. Proceed from unchanged
I006 seq16 to generic R023a arrived-request telemetry, online hybrid-state
reuse, ERS-aware lifecycle scheduling and only profile-justified kernels.

## R023a/R023b In-Flight Hybrid Cache Conclusion

R023a measured the exact I006 seq16 async lifecycle without modifying the
queue or output. All three telemetry/control runs produced the same
Prometheus boundary:

```text
local compute:                456542 tokens
local cache hits:             1962800 tokens
runtime block:                560 tokens
cross-request duplicate:      2240 tokens (0.4906%)
same-request recompute:       0
first-schedule uncached peers:3
```

Therefore native cache is close to the executable lower bound, but not exactly
singleflight. The distinction between scheduled and committed state matters:
`request.num_computed_tokens` advances under `AsyncScheduler` before model
output publishes reusable cache.

R023b tested that distinction directly. Releasing followers only when their
own native cache query reached the target worked for later requests but saved
zero compute. Rechecking early followers caught the real cohort, yet holding
all of them stalled committed cache at `5040/6160`. Every one of the `19`
holds timed out around one second; compute rose by `1120`, cache hits fell by
`1120`, duplicate work rose `50%`, and TPOT p99 reached `19.910ms`.

The technical cause is hybrid aligned-cache publication, not ordinary FCFS
fairness: a follower executes the next state interval and becomes part of the
publication chain. A queue overlay cannot attach unpublished attention KV and
Mamba recurrent state. Removing at most `0.4906%` compute would require an
invasive zero-copy, atomic in-flight state handoff. Reject that work at the
current benefit bound and retain native I006.

## R023c-R023h Integration Closure

R023 closed the remaining generic post-Attempt13 hypotheses without producing
a new integrated candidate.

Native hybrid-state retention is not the current bottleneck. Every one of the
100 later-wave requests had its full shared prefix at first schedule, actual
prefill exactly matched unique suffix work, and worst-active occupancy was
only `32.83%`. Exact renderer reuse is also absent (`0/120` hits); continuation
tokenization is a different problem and remains profile-gated.

The two scheduler overlays prove that cheap policy execution is not enough:

```text
deadline scheduler ERS / midpoint: 0.567834 / 0.570884
deadline fast count / midpoint:    54 / 66
burst scheduler ERS / midpoint:    0.567317 / 0.568643
burst throughput / midpoint:       774.236 / 797.427 tok/s
```

Both kept compute at `456,542`; they rearranged latency rather than removing
work. The official squared score cannot be improved by starving requests whose
TTFT component is already lost when that intervention also delays still-
recoverable requests or decode.

Single-storage FP8 tied weights establish another important boundary. Static
saving of `484-485MiB` did not increase runtime KV capacity because projection
workspace and graph/runtime allocations consumed it. Per-row/per-tensor TPOT
p99 was `21.480/20.553ms`; reject storage-only optimization without runtime
workspace evidence.

The closing profile audit leaves no untested hotspot above the opening gate.
GEMM, full attention and target-H200 GDN prefill already route to CUTLASS,
FlashInfer and the supported backend; tested GDN decode/state and norm-fusion
alternatives are negative. Practical conclusion: preserve I006 seq16 and do
not write another custom kernel until fresh profiling changes the bound.

## DR07 Cache Residency And Allocator Conclusion

The retained-prefix GPU idle time is not explained by physical cache pressure
or native cache-control cost. A direct observer timed 15 native methods while
delegating their behavior unchanged. Its A/B/A remained within `-0.381%` ERS
of the native midpoint and preserved exact compute/cache totals.

```text
cache-control / summed engine TTFT: 0.253%
schedule + update / runtime:        2.075%
lookup / allocate / schedule p99:   101.063 / 51.697 / 475.189us
peak physical pool use:             25.323%
minimum free blocks:                2026 / 2713
true eviction / alloc failure / preemption: 0 / 0 / 0
```

The `1260` maximum hybrid null references describe logical Mamba alignment,
not 1260 unavailable physical pages. The pool retained large headroom while
later-wave hit rate rose to `87.97%`. Pinning, allocator replacement, block
layout tuning, sparse checkpoints and a C++ scheduler therefore have no
profile-backed benefit under the current workload. The next branch is a
capability-first precision audit, not another memory-policy implementation.

## DR16 Quantization And Precision Conclusion

INT4 storage is not itself a serving optimization. The exact checkpoint audit
shows a clean W4A16 G128 shape candidate covering `99.9141%` of text-linear
parameters and saving `632.438MiB` versus current FP8. But DR07 shows no cache
pressure, so the candidate must win the actual Hopper GEMM path.

DR02 gives the performance threshold: only `51.452%` of decode CUDA time is in
the affected FP8 CUTLASS core. The W4 kernel therefore needs
`1.040x/1.102x/1.215x` speed over FP8 for `2%/5%/10%` total decode gain. The
tied BF16 head remains `29.132%` and cannot be silently included: Qwen3.5
vLLM 0.22.1 does not pass quantization into the tied embedding object, while
the official W4 embedding kernel performs input gather/dequant rather than
output logits.

Practical conclusion: park GPTQ/compressed-tensors W4A16 until a reproducible
converter and exact-image Marlin resolution are both proven. Do not benchmark
or package online dense INT8, generic AWQ, BitsAndBytes, NVFP4-on-H200 or
quantized Mamba state from the current evidence. I007 remains the already-
measured output-head watchlist; it is not evidence that whole-model W4 wins.

## DR20 Continuation Cache Integration Conclusion

Frontend token reuse is a real but small performance mechanism. The hardened
DR03b cache independently repeated ERS `0.614390` versus I006 control midpoint
`0.609855` (`+0.7436%`), TPOT p99 `17.556ms`, `120/120`, and `99/100`
continuation coverage with zero unsafe fallback. It reduces tokenizer work; it
does not reduce model prefill/KV compute.

Quality attribution is the limiting gate. The first fresh panel had control
process drift `19/100` and candidate drift `12/100`; its process-noise analysis
found 74 jointly stable cases and zero candidate-specific changes. A second
panel was pre-registered with a zero-change requirement. It found 80 jointly
stable cases and two stable candidate-specific changes, so the candidate failed
without changing the threshold after measurement. All runs retained penalty
proxy `1`, and zero globally stable changes across 67 cases argue against a
systematic accuracy loss, but do not prove the inactive path harmless enough to
ship.

Practical conclusion: retain DR03b as performance-positive research, exclude
it from I006/R024, and do not rerun unchanged. Reopen only if a new causal
mechanism removes process/output divergence and passes the same zero-change
attribution guard.

## R024 Conservative Freeze Conclusion

The final local gate should aggregate evidence, not choose the best repeat.
Across four exact-I006 controls, every run completed `120/120`, local ERS
ranged `0.609243-0.621927`, startup stayed at or below `107s`, and all resource
and single-owner contracts passed. Three controls met TPOT p99 `<=18ms`; the
fourth reached `18.511ms`, so decode-tail variance remains real.

Four fresh I006 GPQA controls completed `100/100` at accuracy `0.35-0.39` and
all retained penalty proxy `1`. Prediction variability across fresh FP8
processes remains, but the complete H200 panel independently reports accuracy
drop `0`, penalty `1` and zero failures.

Practical conclusion: freeze the exact I006 seq16 Compose and image as both
performance candidate and rollback. The local result does not justify tuning
away the outlier or merging a rejected primitive. R025 should be an H200
decision ladder driven by one named uncertainty, not another broad search.

## R025 Score Sprint Conclusion

The useful active-set boundary moved from seq16 to seq19 only after comparing
request-level ERS rather than percentiles alone. I007 seq19 scored `0.610256`
versus selected I006 control `0.609243`, increased TTFT-under-ceiling count
from `77` to `84`, and kept its TPOT component at `0.999995`. Seq20 preserved
the same fast count but pushed TPOT p99 above 20ms and lost ERS, so more
concurrency is not automatically better.

Stronger synthetic readiness warmup did not transfer. It lost on both I006 and
seq19 and did not create another fast-TTFT request. Retain the smaller I005
warmup and reject long-prefill/full-decode/handoff priming as a combined
technique.

The seq19 quality run completed `100/100`, accuracy `0.34`, drop proxy `0.06`
and penalty proxy `1`. Attempt 14 then provided the required H200 attribution.
It scored `46.86`, increased passed SLO `16 -> 19`, and improved I006 TTFT
p50/p95 `4783/6800 -> 4295/6586ms`; therefore seq19 is a real TTFT-tail
mechanism. TBT median regressed `16 -> 19ms`, however, and the complete score
fell `4.73`. Accuracy drop `2` retained penalty `1`, so it does not explain the
score loss.

Practical conclusion at the Attempt 14 checkpoint: do not discard seq19 and do
not call the submission globally bad. Retain its TTFT benefit, keep I006 seq16
as complete champion, and evaluate the R026b graph lattice. The measured R026
answer follows below.

## R026 Decode Graph Lattice Conclusion

Static padding exposure did not translate directly into decode speed. Graph11
first gained `0.005198` ERS against an earlier process, but its repeat at
`0.624805` was almost tied with a later warm control at `0.624707`. The paired
edge was only `+0.000097`, W/T/L `34/28/58`, and TPOT p99 worsened
`18.433 -> 18.519ms`. This is process-order sensitivity, not strong evidence of
a decode optimization.

Adding graph6 was clearly negative: ERS fell to `0.623326`, W/T/L was
`5/28/87`, and TPOT p99 rose to `18.708ms`. The fail-closed protocol correctly
skipped graph13. Graph18 remains parked because its static ceiling is only
`0.196%`.

Practical conclusion: preserve seq19's H200 TTFT mechanism, but do not claim
the graph lattice repairs TBT locally. Graph11 is only a low-confidence,
one-variable H200 calibration artifact. Reject graph6/13 and keep I006 seq16
at score `51.59` as the complete champion.

## R027 Seq18 Interpolation Conclusion

Active-set interpolation produced the decode improvement that graph-set-only
tuning did not. Against the mean of fresh seq19 controls, seq18 reduced TPOT
p95 by `0.353ms` and p99 by `0.530ms`, while throughput changed only
`-0.138%`. Every request remained at or below 20ms TPOT and all three cells
completed `120/120` with startup `97-98s`.

The cost is explicit: fast TTFT fell `92 -> 91`, TTFT p95 increased
`204.0ms` (`7.71%`), and local ERS fell `0.002500` (`0.400%`). TTFT p50 rose
only `7.2ms`, so the loss is concentrated in the tail rather than a broad
prefill regression.

Portal attribution is now complete. Attempt 16 scored `47.04`, retained
`19/120` SLO passes, improved TBT `19 -> 18ms` and accuracy drop `2 -> 0`
versus Attempt 14, but TTFT p50/p95 worsened `4295/6586 -> 4407/8799ms`.
Score rose only `0.18` and remained `4.55` below I006 seq16.

Practical conclusion: the local decode direction was real, but the RTX TTFT
tradeoff materially underpredicted H200 tail loss. Reject seq18 as a complete
candidate, close active-set interpolation and keep I006 seq16 `51.59` as the
target champion. Do not submit graph11 Attempt 15 without new evidence because
its local effect was nearly neutral and slightly decode-negative.

## R028 Prefill Budget Score-Shape Conclusion

A better percentile is not necessarily a better ERS distribution. On exact
public I006 seq16, fixed prefill budget `4096` increased TTFT-under-ceiling
count `78 -> 84`, reduced TTFT p95 `3326.4 -> 3136.7ms`, reduced TPOT p95/p99
`16.315/17.705 -> 15.180/16.539ms`, and increased throughput
`801.228 -> 809.821 tok/s`. Nevertheless, ERS fell
`0.610570 -> 0.601842` (`-1.43%`).

The request-level decomposition resolves the apparent contradiction. The
`<=500ms` cohort fell from a control midpoint of `22.5` to `18`, mean squared
TTFT score fell `0.221140 -> 0.203685`, and all six waves lost request score.
The six extra requests moved just below `1500ms`, where their score gain is
small; slowing already-fast requests is much more expensive under the squared
curve. TPOT score was already `1.0` for candidate and controls, so lower raw
TPOT did not improve ERS.

R030 corrected this telemetry interpretation. Fixed `4096` reduced queue and
decode wall sums `15.98%/5.67%` and increased scheduled-to-first-token prefill
wall sum `48.14%`; that phase metric is not GPU compute. Raw prefix lookup hits
fell `33.87%`, but queries fell `34.54%` and lookup hit rate rose `0.800pp`.
One-per-request cache attribution fell only `0.143%`, with `2,800` more local-
compute tokens. This is a scheduler/interleaving and score-shape tradeoff, not
a one-third cache-reuse collapse, OOM, preemption or eager fallback failure.

Practical conclusion: reject fixed `4096` on I006 and retain the runtime's
resolved/default `2048`. Optimize the per-request score shape, especially the
very-fast cohort, rather than maximizing count under the hard ceiling or raw
throughput. Do not submit this branch or reopen fixed `16384`.

## R029 Dynamic Budget Opportunity Conclusion

Tree-Gated classification cannot manufacture a large serving opportunity when
the underlying counterfactual is small. Choosing the better observed budget
for each request independently gives an optimistic ERS `0.613358` versus
control `0.610570`, only `+0.457%`. Requiring candidate 4096 to beat both
controls lowers the robust gain to `+0.351%`. This already assumes semantics
vLLM does not expose: one budget per request instead of one budget per
scheduler iteration.

Using only legal runtime features, the full-data stump and depth-2 tree fitted
positive gains of `0.000779` and `0.001523`. Leave-one-wave-out reduced the
stump to `+0.000301` midpoint gain and only `+0.000046` robust gain. Depth-2
reversed to `-0.000224` midpoint and `-0.000764` robust gain. It selected
different roots across folds and was positive in only two waves.

The important distinction is between classification accuracy and causal
serving value. A model may identify some 4096-positive observations in two
complete global runs, but changing the live global budget changes queueing,
cache publication and every request's latency. Per-request replay therefore
is an upper bound, not an implementation simulation.

Practical conclusion: do not retrain a larger XGBoost/C++ Tree-Gated model for
this budget decision. More model capacity would fit the 120 observations more
tightly and worsen the demonstrated generalization problem. Close dynamic
`2048/4096` switching without GPU and seek a primitive that lowers actual
prefill/cache work.

## R030 Prefix Metric Semantics Conclusion

Counter names are not sufficient evidence of causal work. In exact I006,
`prefix_cache_queries_total` and `prefix_cache_hits_total` are updated by
`get_computed_blocks()` before slot allocation. A waiting request whose
allocation fails can be looked up and counted again on a later scheduler
iteration. Their raw magnitudes therefore include repeated lookup attempts.

For workload attribution, subtract prebenchmark from final counters and use:

- matching query/hit rate for lookup behavior;
- `prompt_tokens_by_source_total{source="local_cache_hit"}` for actual request
  cache attribution;
- `prompt_tokens_by_source_total{source="local_compute"}` for actual request
  prompt compute;
- request phase timers only as wall durations unless a GPU profiler proves
  kernel time separately.

Applied to R028, lookup hit rate improved `0.800pp`, while actual local-cache
tokens fell only `2,800` (`0.143%`) and local-compute tokens rose `2,800`
(`0.613%`). This correction leaves ERS, request-score decomposition and frozen
gates unchanged, so fixed `4096` remains rejected and R029 remains closed.
The practical action is to prevent a false cache-loss research branch, not to
revive the candidate.

## R031 First-Token Lifecycle Attribution Conclusion

Queue time and prefill-phase wall time must be separated at the first schedule
boundary. R031 used a measurement-only scheduler overlay and a forward plus
reverse exact-I006 bracket, giving three independent processes per budget.
The overlay delegated all scheduling and state mutation to the stock path and
recorded no prompt text, token IDs, outputs or request identity.

Across the six runs, fixed `4096` changed mean ERS `0.608403 -> 0.597467`
(`-1.80%`). Mean queue time improved by `138.716ms`, but mean time from first
schedule to first token worsened by `122.927ms`. The larger chunk reduced
prefill lifecycle steps `3.54 -> 2.65`, yet each request shared those steps
with about `3,025.714` additional peer-prefill tokens. Actual request-local
prompt compute rose by only `23.333` tokens, confirming that batch composition
and first-token service granularity, rather than lost cache reuse, dominate.

The service penalty appeared in all six waves. Total score lost in five; wave
2 won because its queue saving exceeded its smaller service penalty. The
pre-registered all-six-wave gate therefore fails. Keep `2048`, do not build a
dynamic budget scheduler, and do not create Attempt 17. A final-chunk/service
isolation idea may proceed only as an offline opportunity audit with
leave-one-wave-out evidence before any implementation.

## R032 Final-Chunk Isolation Opportunity Conclusion

An ideal phase splice can expose a ceiling without proving that a scheduler
can realize it. R032 crossed all three `2048` processes with all three `4096`
processes and subtracted only a net fraction of positive service excess. This
retained candidate queue/frontend/TPOT behavior and made no claim about exact
final-step peer work.

At `50%` recovery the mean relative ERS gain was `+0.775%`, but the worst pair
was `-0.497%` and only `6/9` pairs won. Waves 1 and 5 remained negative; the
`<=500ms` cohort lost `1.019%`. All process pairs became positive at `75%`,
but one pair needed `66.85%` net recovery just to exceed `+0.5%` ERS. That is
too little margin for deferred peer work, queue debt and scheduler overhead.

Practical conclusion: close final-chunk isolation before step telemetry or
GPU implementation. Do not use the optimistic `100%` gain (`+3.804%` mean)
as a performance prediction. No R032b or Attempt 17 is justified.

## R033 DR03b Score-First Requalification Conclusion

The previous DR20 rejection used a stricter research attribution rule than the
official BTC quality gate. R033 corrected that distinction: five historical
candidate GPQA processes scored `0.34-0.36`, all retain penalty `1`, cache hits
were zero on GPQA, and exact output drift is therefore diagnostic rather than
an official failure.

Historical performance still could not promote DR03b by itself because its
controls used a different image. R033 removed that confound with one frozen
same-image off/on/off bracket. Candidate ERS was `0.612869` versus control
midpoint `0.614450` (`-0.257%`), TTFT-under-ceiling count was `81` versus
`83/83`, W/T/L was `32/35/53`, and only three waves were positive. The score
delta was entirely TTFT (`-0.001581`); TPOT score remained saturated at `1.0`.

The cache itself behaved correctly: `99` continuation hits, `1,846,706` saved
tokenizer tokens, 64-entry bound, and zero kwargs/reconstruction fallback.
Therefore the technical conclusion is narrow: tokenizer work reduction is
real, but its end-to-end score gain is below process/scheduling variance on
this pipeline. Close unchanged DR03b. The conditional GPQA confirmation was
skipped after performance failed, not counted as a quality failure.

## R034 GPTQ W4A16 / Marlin Conclusion

R034 separates three claims that must not be conflated: checkpoint
compressibility, model quality and serving speed. The exact Qwen3.5 hybrid
artifact is compressible: GPTQModel quantized `150` safe linears, the tensor
payload is about `2.397GB`, tokenizer and manifest hashes remain exact, and
vLLM loads it through native GPTQ Marlin. Three GPQA runs scored
`37/37/35`, so the official penalty proxy remains `1` with `0.02` process
span.

Serving speed is negative. In a frozen same-image FP8/GPTQ/FP8 bracket,
candidate ERS was `0.542724` versus control midpoint `0.609622` (`-10.974%`).
TTFT p50/p95 rose to `1978/4308ms`; TPOT p95/p99 rose to
`22.209/23.213ms`; throughput fell `804.989 -> 791.048 tok/s`; W/T/L was
`0/37/83`; and all six waves lost. Control spread was only `0.534%`, so this
is not session drift.

The memory trade also explains why compression did not help. W4 increased
available GPU blocks `2702.5 -> 2827` (`+4.607%`) and reduced measured peak
VRAM about `128MiB`, but prior cache audits already showed no eviction,
allocation failure or capacity pressure. Meanwhile sampled GPU utilization
rose `44.9% -> 86.1%` and power `185.8W -> 234.0W`. On this RTX 5080/vLLM
path, Marlin unpack/dequant compute loses to native FP8 CUTLASS; unused KV
headroom cannot recover that cost.

Practical conclusion: reject all-core GPTQ/AWQ W4A16 on this runtime and do
not create Attempt 17. H200-specific INT4/NVFP4 is a separate hypothesis only
if the target image exposes a distinct native kernel and a target-hardware
microbenchmark proves speed before packaging.

## R035 Timed JIT Accounting Conclusion

R035 found a measurement-label error, not a runtime regression. In every R034
cell, `server_prebenchmark.log` is an exact byte prefix of `server.log`. The
prefix contains five JIT warnings, followed by the packaged synthetic-warmup
report and public relay readiness. The suffix containing timed benchmark
traffic has zero JIT warnings.

The five kernels belong to KV block zeroing/slot mapping and the hybrid
Mamba/GDN state and prefill paths. They are already compiled by legal,
trace-independent I006 startup warmup. The analyzer now reports
`prebenchmark/timed = 5/0` and fails closed if the snapshot is not an exact
prefix.

Practical conclusion: do not expand warmup or run another GPU bracket for
these kernels. R035 creates no speedup and changes no R034 score decision; it
removes a false optimization target.

## R036 H200 Transfer Residual Conclusion

R036 proves why local total ERS is insufficient for H200 candidate selection.
Nine immutable local runs all passed `120/120` and recomputed exactly, yet
local score direction disagreed with Portal for active-set 20 to 16, the FP8
tied head, and active-set 19 to 18. The useful transfer signal is component
level, not total score: reducing active set improved H200 TBT and score twice
despite non-improving TTFT support and TTFT p50/p95.

The official score formula also gives useful bounds without request-level
Portal logs. I006 seq16 score `51.59` and `16/120` requests inside the TTFT
ceiling imply only `1.590..6.667` TTFT points and `44.923..50.000` TPOT points.
The absolute optimistic score ceiling without increasing TTFT support is
`56.666667`. Score 70 needs at least `48/120` supported requests and score 100
needs all 120.

Median TBT is not mean TPOT score. Attempts 09, 11 seq20, 14 and 16 report TBT
median at or below `20ms` while total score is below 50. Since TTFT points are
nonnegative, those TPOT components are provably unsaturated. This explains
why a one-millisecond H200 decode change can outweigh better TTFT tails.

Practical conclusion: preserve I006 seq16 and inspect exact-image H200 decode
capabilities offline before another GPU run. Decode repair can provide a small
score step, but a real breakthrough requires a new causal primitive that
raises TTFT support while preserving the TPOT distribution. Closed scalar
active-set, graph, W4, budget, warmup and exact-trace branches do not qualify.

R037 completes that exact-image inspection. vLLM 0.22.1 exposes
`VLLM_SSM_CONV_STATE_LAYOUT=DS`, and Qwen3.5 GDN consumes the resulting
dimension-first state directly. It is semantically safe for I006 because the
control does not use speculative rollback. The measured ceiling is still too
small: causal convolution accounts for only `1.296%` of pure-decode GPU time
and `1.196%` of cold-prefill GPU time. Even eliminating the whole kernel would
only move a 16ms decode interval to about 15.79ms, where the TPOT component is
already at its floor. MRV2 is larger but incompatible with the retained
Mamba `align` cache; disabling prefix caching would turn `456,542` measured
prompt-compute tokens into up to `2,419,342` (`5.299x`). Close both easy paths
without GPU. The next score-bearing design must increase requests receiving a
first token before 1500ms, not optimize another one-percent kernel.

## R038 First-Token Staging Opportunity Conclusion

R038 tests the user's one-model, two-policy idea at the request-lifecycle
boundary rather than by loading two engines. The final simulator is causal at
the declared abstraction: staged prefills share one serial lane, first-token
events update decode occupancy, parked time is charged to TPOT, and overlapping
prefill time is charged to active decode. It consumes only six immutable R031
timing panels and no prompt, token or output content.

The guarded point is `slots8`, queue age `>=500ms`, 75% queue-delay removal
and 10% decode interference. It gains `+0.016036` mean ERS, has
`+0.013850` worst-process gain, stays positive in all `36` twenty-request
waves, and models TPOT p95/p99 no worse than `18.183/19.032ms`. Mean fast-TTFT
count rises by `13.7` requests.

The mechanism is fragile in a technically useful way. At 50% queue removal,
best mean gain is only `+0.007925`, below the process-drift gate. At 25%
decode interference, mean ERS remains positive but TPOT p95/p99 rises to
`19.677/21.634ms`, outside the declared guard. The implementation therefore
cannot be a generic queue reorder or global prefill-budget switch.

Practical conclusion: open only the gap-aware design contract. One native
engine may expose 24 state slots while keeping 16 logical decoders and at most
eight serial staged/parked requests. Decode must be scheduled first; staged
chunks may consume measured slack only; all KV and Mamba state stays in the
same engine. Runtime telemetry must fail closed to I006 below 75% queue gain
or above the TPOT guard. R038 itself creates no measured speedup, engine,
image, Compose or Attempt 17.

## R039 Static Scheduler Conclusion

The R038 mechanism is representable as a small fail-closed policy, but that is
not evidence that vLLM can safely expose the required lifecycle state. R039's
policy keeps decode first, stages at most one bounded chunk, prioritizes parked
state, and gates staging on measured queue gain, interference and TPOT. It
rejects non-finite telemetry and interference outside `[0,1]` rather than
guessing.

The policy has no engine, network or payload boundary violation in the AST
audit and passes all eleven deterministic tests. The future adapter must leave
native block management, prefix caching and hybrid Mamba state ownership in
place. Runtime remains unmeasured because the local NVIDIA driver is broken;
therefore R039 is a protocol checkpoint, not a performance improvement or
submission candidate.

## R040 Parked-State Runtime Conclusion

R040 resolves the hybrid-state uncertainty left by R039. On the tested vLLM
0.22.1 path, a post-first-token request can leave the scheduler running set and
later resume from retained Transformer KV and Mamba recurrent state. Direct
hold and first-token swap both completed `20/20` exact-length outputs with zero
preemption, error, OOM or continuity failure.

That mechanism does not make the policy useful. At matched local GPU
reservation `0.72`, first-token swap raises TTFT component
`0.043405 -> 0.046069` but lowers TPOT component
`0.888938 -> 0.862692`; ERS proxy falls `0.466172 -> 0.454380`. TTFT support
stays `5/20`. Two staged requests gain about one second but remain at
`4323/4583ms`, while TPOT p95 reaches `26.491ms`.

Practical conclusion: the offline R038 model overestimated actionable queue
removal for requests already outside seq16. Close parking policies without a
full trace. Future TTFT work must reduce scheduled-to-first-token service
inside the first active cohort, and must model the squared score loss to the
already-fast requests before implementation.

## R041 First-Wave Service And FP8 Backend Conclusion

The first active cohort does not become slower because later requests require
more scheduled service. Across R031 controls, queue span is `2510.046ms` while
service span is `-117.523ms`; request index and TTFT correlate above
`0.99985`. A robust modeled gain requires `11.5%` queue reduction, `17.5%`
service reduction, or `6.5%` whole-prefill reduction.

The exact image supports serialized static FP8 and that scale layout makes
FlashInfer executable. This is a genuine capability change, but not a speedup.
On all 570 weighted Qwen cold-prefill GEMMs, FlashInfer is `8.762%` slower
than CUTLASS and PyTorch row-wise scaled-MM is `3.954%` slower. Static CUTLASS
removes only `1.319%` in the standalone quant-plus-MM path. Backend numerical
agreement and finite-output checks pass.

Practical conclusion: keep online FP8 and native CUTLASS. Do not spend model
conversion, calibration or full-trace budget on static FP8 in this image. A
future H200-only reopening requires a changed backend implementation and at
least `9.467%` same-shape GEMM improvement, not a feature-availability claim.

## R042 Conditional Budget Conclusion

The apparent positive 4096 cohort is real only as a weak mean effect. Wave 2
is positive in both order brackets, but the reverse effect is only
`+0.001507`, six of nine process pairs agree, and the lower-quartile effect is
`-0.008325`. The other five waves lose in mean direction.

An oracle that illegally chooses a budget independently for each request can
gain `+0.002348` ERS. The scheduler-compatible oracle can choose only a global
budget per arrival burst and gains `+0.001128`; its robust gain is zero. That
upper bound is below both the `0.005` opening gate and process noise
`0.004843`, before accounting for classifier or mutation overhead.

Practical conclusion: close Tree-Gated/global conditional budget selection,
not merely the current model. More training data or C++ inference cannot add
value to an action whose robust executable oracle is zero. Preserve fixed
2048 and move to a distinct prefill compute/fusion mechanism.

## R043 Fused Reduction Conclusion

vLLM's native fused RMSNorm plus dynamic FP8 op is a real callable capability,
but it is not the production optimum for this Qwen workload. It improves
M1/M20 versus two eager raw kernels, then scales poorly: M1680 is
`175.056%` slower and the weighted four-long/one-tail prefill path is
`156.311%` slower.

The decode comparison also demonstrates a methodology rule. The apparent
M20 gain uses an artificial two-launch control. R012's compiled graph already
generates one Triton RMSNorm/quant reduction averaging `2.832us`, which beats
the native fused op's `3.822us` by about 35%. Category names alone did not
reveal that existing fusion.

Practical conclusion: retain Inductor reductions and close matcher/manual-op
integration. Future kernel work must benchmark against the exact compiled
serving kernel and pass an Amdahl gate before engine modification.

## R044 Remaining Hotspot Conclusion

Full attention and hybrid GDN/Mamba are the only remaining cold-prefill
categories above the opening share threshold, at `11.092%` and `9.945%`.
However, the exact H200 target already resolves to the specialized path:
FlashAttention-3 for FP8-KV full attention and FlashInfer for Qwen GDN
prefill.

All eight alternatives fail at least one hard gate. Forced preferred backends
are not distinct, forced Triton selects fallback behavior with negative prior
evidence, CuteDSL is not executable on SM90, and no custom implementation has
a same-path benchmark. Thus there is no causal GPU candidate despite the
categories being material.

Practical conclusion: retain backend `auto`. Do not turn an Amdahl ceiling
into an optimization claim. A future custom attention/GDN branch starts only
with implemented code that beats the exact compiled target on representative
shapes and preserves TPOT.

## Part 2 R001 LFM2.5 One-Shot Requalification Conclusion

The active BTC workload uses `LiquidAI/LFM2.5-1.2B-Instruct`, not the legacy
Qwen model. Native vLLM 0.22.1 identifies it as a hybrid inner-state model and
selects the production attention/short-convolution paths without an external
framework or custom kernel.

Online `fp8_per_tensor` weights are the only large measured first-shot gain.
On the clean 120-request Poisson proxy, BF16 scored `50.806` while FP8
per-tensor seq8 scored `67.715`; TTFT mean fell `120.872 -> 84.222ms` and TPOT
mean fell `3.694 -> 2.514ms`. The paired 100-question regression proxy showed
zero BF16-correct to FP8-wrong transitions. This is not a substitute for the
official full GPQA post-online run, but it clears the local promotion gate.

FP8 KV is not selected. vLLM disables dynamic KV-scale calibration for this
hybrid model because recurrent state is uninitialized during calibration, so
the path falls back to scale `1.0`. Its small local TPOT gain does not justify
the larger answer-change surface or post-online quality risk. Mamba state
stays BF16. `mamba-cache-mode=all` is also unavailable for this model in the
official image and automatically falls back to `align`.

The clean scalar sweep is flat: seq6/seq8/seq10 and prefill budgets
4096/8192 differ by roughly one point, while budget16384 worsens the TTFT tail.
The first Portal artifact therefore uses seq8 and budget8192. This avoids
splitting a prompt that slightly exceeds 4096 after chat templating and keeps
two-prompt prefill capacity for Poisson bursts. Prefix caching remains enabled;
an exact duplicate probe produced 3488 cached tokens, proving the hybrid cache
path works without any trace preload.

Practical conclusion: submit the official-image FP8 per-tensor candidate with
BF16 KV/Mamba state, interactivity mode, seq8, budget8192 and target memory
fraction `0.95`. Do not carry old Qwen sweet spots, FP8 KV, exact-trace warmup,
custom proxies or unsupported Mamba modes into Part 2 Attempt 01.

Final exact-config validation passed under `3 CPU / 8 GiB RAM`: startup took
`90s`, then a distinct-seed 24-request Poisson smoke scored `66.275`, with
TTFT p50/p95 `77.018/287.859ms`, TPOT p50/p95 `2.325/2.951ms`, zero failures
and every request below both latency ceilings. All observed first-use Triton
JIT occurred in warmup before timed traffic. The frozen Compose is `1099`
bytes with SHA-256
`4c6425ceae288fc7155e08e692565cac98f739f50e15676b6a8804a02f209c0e`.

A third-seed full repeat on that exact final config scored `67.086` over
`120/120` successful requests. TTFT p50/p95 was `76.908/137.953ms` and TPOT
p50/p95 was `2.548/3.283ms`; every request remained below both ceilings and
the run emitted no JIT, fallback, OOM or runtime error. Together with the
earlier `67.715` clean run, this confirms the selection direction rather than
a single favorable process.

Portal Attempt 01 scored `56.53` over all `330` graded requests after `90`
primer requests, with TTFT p50/p95 `73/116ms`, TBT median `4ms` and zero
failures. TTFT transferred well; the local-to-Portal score gap is decode-side.
At the current operating point, the score derivative makes a `1ms` TPOT
reduction worth about `34ms` of TTFT, so the next round should deliberately
trade TTFT headroom for decode protection. The Portal accuracy fields are
online placeholders, not GPQA evidence.

## Part 2 R002 Portal-Calibrated Decode Conclusion

Round 002 isolates the Portal gap as decode-side and promotes three compatible
changes over Attempt 01: FP8 E4M3 attention KV, async scheduling and a 16384
prefill token budget. BF16 Mamba state, seq8, interactivity, prefix caching,
chunked prefill, context 32768 and automatic kernel selection remain fixed.

FP8 E4M3 KV repeated its local performance gain and passed the paired
100-question drift gate at `21/100` versus `21/100`, with zero
control-correct-to-candidate-wrong transitions. E5M2 KV, FP16 Mamba state,
seq10, throughput mode, explicit graph sizes, compile size 4096, disabled
prefix cache and higher local cache reservation were all rejected.

The final b16384 candidate scored `67.879` and `68.740` over two cold full
420-request processes, with `660/660` graded successes and mean `68.310`.
The former b8192 incumbent mean was `67.261`. The packaged Attempt 02 keeps
official H200 memory `.95`; local `.86` could not start only because the RTX
5080 display workload left `12.95 GiB` free versus `13.28 GiB` requested.

Practical conclusion: submit the official-image b16384 candidate with FP8 E4M3
KV and async scheduling. Do not add custom warmup, graph overrides, alternate
backends or trace-dependent logic; none passed the current evidence gates.

## Part 2 R002a Turn-Semantics Conclusion

The public trace gives `3997-4000` input tokens on every one of six turns and
exposes no state/session handle. Treating those values as newly appended tokens
creates a sixth request near 25k tokens, contrary to the stated approximately
4k input context. Fixed-size or rolling 4k prompts are therefore the stronger
contract interpretation, though only the hidden driver can prove it.

The distinction changes the correct cache policy. Exact append-only replay
produced about `71.20%` cache-hit tokens; disabling hybrid prefix caching then
collapsed the 90-graded proxy from `57.464` to `20.736`. Conversely, a real-time
fixed 3k-prefix diagnostic issued `719,796` cache queries and recorded zero hit
tokens. Mamba `align` cannot reuse an arbitrary repeated prefix unless its end
is also a retained scheduler-step checkpoint.

On token-identical fixed-size prompts, no-prefix won both 90-request screens
and the full 330-request A/B. Full ERS improved `68.458 -> 72.068`; TTFT mean
fell `85.283 -> 70.611ms`, TPOT stayed effectively flat, every request passed
both ceilings, and failures remained zero. The gain is concentrated in 74
low-contention requests below `40ms` TTFT; the prefix control had none below
`60ms`.

Practical conclusion: retain prefix-enabled Attempt 02 as an append-only hedge
and submit no-prefix Attempt 03 as the more likely fixed-size-semantics hedge.
Do not combine these mutually exclusive policies or infer a universal cache
win from either workload. Budget 20480 was neutral and seq10 regressed in the
append-only branch.

## Part 2 R002b Fixed-4k No-Prefix Tuning Conclusion

Once prefix bookkeeping is removed, the remaining scalar operating point is
flat. Fixed controls bracketed at `71.836/71.951` points. Against their
`71.894` mean, budget20480 lost `0.273`, no-chunk with its required budget32768
gained only `0.001`, seq6 averaged only `+0.107` across two cold processes, and
throughput mode lost `0.458`.

The invalid no-chunk/b16384 startup is a contract constraint: without chunked
prefill, vLLM requires the token budget to be at least max model length32768.
The valid b32768 run is neutral, so removing chunk bookkeeping does not pay for
the larger scheduling envelope. Seq6's opening gain reversed on repeat.

Practical conclusion: keep no-prefix Attempt 03 at seq8, b16384, chunked
prefill and interactivity. Skip b24576, open no full trace, and create no new
artifact from this scalar sweep.

## Part 2 R002c GPU N-Gram Speculative Decode Conclusion

vLLM 0.22.1's `ngram_gpu` proposer is executable with async scheduling on the
exact LFM2.5 stack, but it is not useful for this workload. A 96-request
diagnostic generated `5820` draft tokens and accepted only `67` (`1.15%`);
the first draft position was accepted in only `40/2910` steps (`1.37%`).

This is an accuracy-of-the-proposer failure, not a draft-coverage failure: two
drafts were generated on every eligible step. Broader unigram lookup therefore
cannot repair it, and a third speculative token only extends a continuation
whose first position is already almost always wrong. The path also forces the
FlashInfer runtime from full-and-piecewise CUDA graphs down to piecewise.

Practical conclusion: reject built-in GPU n-gram speculative decoding before
a performance/full trace. Preserve the no-spec Attempt 03 stack and reopen
speculation only for a proposer with measured representative acceptance high
enough to pay for verification and graph-path overhead.

## Part 2 R002m Decode-Head V2 Conclusion

R002k has exhausted the practical exact output-head opportunity on RTX 5080.
Directly owning the reduction output is lifetime-safe under vLLM async D2H but
gains only `0.011%` weighted GPU time. Removing clone without ownership is
unsafe and bounds the entire remaining overhead at only `0.489%`.

A second transposed BF16 head is slower and costs 256 MiB. Manual GEMV is
competitive only at M1; it rereads the full tied weight per row and becomes
about 2x/4x/8x slower at M2/M4/M8. Keep tensor-core R002k and move subsequent
kernel research to a profile-proven model-layer hotspot rather than output
allocation or weight layout.

## Part 2 Portal A/A Variance Conclusion

The exact R002k artifact scored `60.75` and `58.94` without any Compose, image,
flag or model change. Its two-run mean is `59.845`; observed ERS range is
`1.81`, TTFT-p50 range `11ms`, TTFT-p95 range `1ms`, failure range one, and TBT
median range zero.

This is not a statistical confidence interval, but it is enough to reject
single-run attribution below roughly two points. Retain best score `60.75` for
leaderboard state, mean `59.845` for expected-value planning, and require
repeated local evidence plus a mechanism before creating another Portal file.

## Part 2 R002n ShortConv Projection Conclusion

The exact LFM2.5 model is hybrid in implementation, but its ten ShortConv
layers use ordinary vLLM linear projections. The active FP8 quantization config
does not reach those constructors, leaving a measurable `19.499%` of steady
decode child-kernel time on BF16 projection GEMMs. This is a real optimization
surface distinct from the already-promoted R002k output head.

Passing the quant config to `in_proj` alone is the best current branch. It
reduces fixed-batch child-kernel time and improves two local full-replay ERS
observations, while `out_proj` alone is neutral/negative and both projections
create a clear GPQA proxy regression. The in-only branch is not yet safe to
package: it changes generated sequences materially, misses the 400ms TTFT SLO
once per run, and the local control has a large process envelope. Treat its
positive ERS as an opportunity signal that requires clean-target transfer and
selected-GPQA validation, not as a stable score forecast.

## Part 2 R002o Phase-Aware ShortConv Conclusion

Phase-aware routing can preserve native BF16 prefill while using an independent
FP8 projection during decode in one vLLM model process. The checkpoint can be
loaded once by aliasing parameters before quantization; vLLM then replaces only
the online-FP8 module's weight storage. This mechanism is graph-compatible and
works for the hybrid LFM2.5 ShortConv path.

That mechanism is not itself a performance win. The all-layer path reduced
summed decode child-kernel time by `6.72%`, yet two exact full replays averaged
`53.0663` versus immediate control `55.4200`. It added ten dynamic activation
quantization launches per decode step, about `0.11 GiB` model storage and worse
request-level tails. The quality-neutral two-layer mask saved only `0.845%` in
the profiler.

Practical conclusion: child-kernel sums can identify an opportunity but cannot
authorize integration. Close phase-aware ShortConv FP8 and keep R002k. Future
FP8 work must measure CUDA-event wall time and remove activation-quantization
or launch overhead before full replay; a dual-path wrapper alone is insufficient.

## Part 2 R002p FP8 Activation-Scale Conclusion

The online FP8 weight path is not dominated by activation scaling. Dynamic
per-token quantization contributes only `4.6195%` of clean decode child-kernel
time because TorchInductor already fuses it with adjacent normalization,
residual and activation work. Oracle static microbenchmarks therefore
overstate deployable gain.

All-static production profiling removed 44 reductions but improved complete
batch-8 child-kernel time by only `1.2296%`; separate static cast/clamp work
remained. More importantly, two legal calibration strategies both regressed
the repeated 100-case quality proxy. Decode-aware maxima did not help because
larger ranges trade clipping for coarser FP8 resolution.

Python lifecycle state is also not a graph-native policy boundary. Once vLLM
captures the dynamic branch, changing scheduler state cannot select a static
linear graph, and tensor row counts reflect compile padding rather than active
decode width. Future phase-specific quantization must be represented inside
the captured graph or by separate graph variants. For this workload the
measured ceiling does not justify that complexity; retain dynamic per-token
activation quantization.

## Production Reminder

RTX 5080 local baseline chỉ dùng để:

- phát hiện hướng tối ưu tốt/xấu
- so sánh thay đổi trong cùng một máy
- tránh đẩy cấu hình rõ ràng tệ vào submission

Không dùng RTX 5080 local result làm:

- baseline chính thức
- ước lượng leaderboard
- kết luận cuối cho H200 MiG

## Part 2 R002q Graph-Native Fusion Conclusion

vLLM's built-in fusion switches are not equivalent to eliminating all dynamic
FP8 quantization. On LFM2.5, `fuse_norm_quant` matches five graph patterns and
reduces the visible quant producer count by half, but the complete decode step
improves only `0.597%` because the surrounding graph work is regenerated in a
different family. `fuse_act_quant` has no matches for the active dynamic
per-token path. `fuse_attn_quant` has no matches and changes graph capture
policy, so it is a poor generic knob for this model.

The practical lesson is to gate fusion by complete CUDA-event step time, not
by matcher count or one kernel family. The next meaningful opportunity is the
FP8 scaled-MM backend itself, which accounts for about `51%` of the profiled
decode child-kernel time; R002k remains unchanged until that branch clears the
same drift and quality gates.

## Part 2 R002r FP8 Linear Backend Conclusion

A backend name cannot overcome a scale-layout mismatch. The current
`fp8_per_tensor` policy stores per-tensor FP8 weight scales but dynamically
quantizes activations per token. FlashInfer's per-tensor kernel therefore
cannot instantiate, while Torch falls to a channel-wise implementation that
computes FP32 output and applies scale/dequantization separately. Marlin and
the block-scaled Triton/DeepGEMM families solve different hardware or
representation problems.

On RTX 5080, Torch's raw replacement GEMMs are about `2.36%` faster than the
CUTLASS FP8 family, but the complete decode step is `0.0396%` slower in a
low-drift bracket. Raw GEMM timing is therefore another insufficient proxy:
epilogue and conversion work must be included. Keep automatic/CUTLASS for the
current representation. The next credible linear optimization must either
prove a quality-safe block-scaled representation with an SM90 kernel or fuse
dynamic per-token scaling natively; switching existing backend flags is
closed.

## Part 2 R002aa Sparse ShortConv Checkpoint Conclusion

LFM2's cached-turn alignment step is a state-boundary problem, not a useful
ShortConv-kernel hotspot. A sparse manager can retain only the final crossed
full-block state and final partial tail, preserve align-like cache capacity and
avoid the tiny second model step. Retaining every block boundary is rejected
because it cuts cache capacity from about `1.59M` to `0.53M` tokens.

The implementation is structurally valid and quality-safe on the fixed local
proxy, but performance attribution remains target-dependent. Two clean no-pad
replay pairs win by `+0.8713` ERS on average, while one compositor-contaminated
pair loses and is excluded by the pre-registered idle rule. Pad32 demonstrates
why micro wins are not promotion evidence: it passes two micro repeats and then
loses the full replay.

Packaging must also isolate research instrumentation from serving code. The
first image crashed because an optional profiler was imported without its
environment contract; the repaired image omits that profiler and completes
`420/420`. Promote R002aa only after the exact immutable image runs on H200;
until then R002k remains the production baseline.
