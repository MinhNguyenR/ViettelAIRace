# Research Draft

Version: 1

## Production Philosophy

Benchmark là công cụ xác minh. Production mới là mục tiêu cuối cùng.

Mọi quyết định tối ưu trong dự án đều phải ưu tiên khả năng vận hành thực tế trước khi tối đa hóa điểm benchmark. Không tối ưu cho benchmark. Tối ưu cho workload. Benchmark chỉ dùng để xác minh rằng workload đã được phục vụ tốt hơn.

## Scope

Xây dựng một hệ thống LLM Serving hướng production, tối đa hóa hiệu năng tổng thể dưới workload thực tế thông qua quá trình đo đạc, tối ưu và lặp lại liên tục, đồng thời duy trì chất lượng đầu ra của mô hình.

## Success Metrics

Success Metrics chia thành 8 nhóm.

### I. Latency

Càng thấp càng tốt.

| Metric | Ý nghĩa | Mục tiêu |
| --- | --- | --- |
| TTFT | Time To First Token | xuống |
| TPOT | Time Per Output Token | xuống |
| End-to-End Latency | Tổng thời gian request | xuống |
| P50 Latency | Median | xuống |
| P90 Latency | 90% request | xuống |
| P95 Latency | 95% request | xuống |
| P99 Latency | Tail latency | xuống |

### II. Throughput

Càng cao càng tốt.

| Metric | Ý nghĩa | Mục tiêu |
| --- | --- | --- |
| Requests/sec | Số request mỗi giây | lên |
| Tokens/sec | Throughput token | lên |
| Decode Tokens/sec | Tốc độ decode | lên |
| Prefill Tokens/sec | Tốc độ prefill | lên |

### III. GPU Efficiency

| Metric | Mục tiêu |
| --- | --- |
| GPU Utilization | lên |
| SM Utilization | lên |
| Tensor Core Utilization | lên |
| Memory Bandwidth Utilization | lên |
| HBM Bandwidth | lên |
| GPU Idle Time | xuống |
| CUDA Kernel Efficiency | lên |

### IV. Memory

| Metric | Mục tiêu |
| --- | --- |
| VRAM Usage | ổn định |
| KV Cache Usage | lên trong giới hạn an toàn |
| KV Cache Hit Rate | lên |
| Prefix Cache Hit Rate | lên |
| Memory Fragmentation | xuống |
| OOM Count | xuống |

### V. Scheduler

| Metric | Mục tiêu |
| --- | --- |
| Average Queue Length | xuống |
| Waiting Time | xuống |
| Scheduler Overhead | xuống |
| Batch Utilization | lên |
| Average Batch Size | lên |
| Batch Fill Rate | lên |

### VI. Model Quality

| Metric | Mục tiêu |
| --- | --- |
| Accuracy | lên |
| Exact Match | lên |
| Pass Rate | lên |
| Perplexity | xuống |
| BLEU/ROUGE | lên nếu cần |
| Output Validity | lên |

### VII. Stability

| Metric | Mục tiêu |
| --- | --- |
| Crash Count | xuống |
| Timeout Rate | xuống |
| Error Rate | xuống |
| Retry Rate | xuống |
| Success Rate | lên |
| Availability | lên |

### VIII. Cost Efficiency

| Metric | Mục tiêu |
| --- | --- |
| GPU Hour Cost | xuống |
| Cost / 1M Tokens | xuống |
| Energy Consumption | xuống |
| Tokens / Watt | lên |
| Requests / Dollar | lên |

Một tối ưu chỉ được coi là thành công khi cải thiện ít nhất một chỉ số mục tiêu mà không làm suy giảm đáng kể các chỉ số còn lại. Mọi thay đổi đều phải được xác nhận bằng benchmark có thể tái lập.

## 4. Production Assumptions

- Hardware Assumptions
- Software Assumptions
- Model Assumptions
- Workload Assumptions
- Deployment Assumptions

## 5. Constraints

- Không sửa architecture model.
- Không fine-tune nếu BTC không cho phép.
- Chỉ tối ưu inference.
- Chỉ dùng GPU BTC cung cấp.
- Thời gian cuộc thi giới hạn.

## 6. Threat Model

Điều gì có thể khiến benchmark đẹp nhưng production tệ:

- Prompt quá ngắn.
- Prompt quá dài.
- Cold Start.
- Burst Request.
- Memory Pressure.
- Tail Latency.
- Queue Starvation.
- Fragmentation.
- Scheduler Bias.

## 7. Production Workload Model

User Persona:

- Chat
- Coding
- Agent
- Long Context
- Mixed

Prompt Distribution:

- Short
- Medium
- Long

Decode Distribution:

- Short
- Medium
- Long

Concurrency Distribution:

- 1
- 4
- 8
- 16
- 32
- 64
- 128
- 256
- 512

Request Pattern:

- Continuous
- Burst
- Idle
- Mixed

## 8. Benchmark Methodology

- Environment
- Hardware
- Driver
- CUDA
- Framework
- Fixed Parameters
- Variable Parameters

## 9. Metrics Collection

- Nsight Systems
- Nsight Compute
- nvidia-smi
- DCGM
- Prometheus
- Grafana
- TensorRT profiler
- vLLM metrics

## 10. Baseline

Không tối ưu gì. Benchmark nguyên bản.

## 11. Bottleneck Analysis

Tìm bottleneck lớn nhất:

```text
CPU
GPU
Memory
KV
Scheduler
Batch
Network
Storage
Framework
Kernel
Model
```

## 12. Optimization Candidate Pool

Đây không phải roadmap. Chỉ là danh sách.

Runtime:

- TensorRT-LLM
- vLLM
- SGLang

Scheduler:

- Continuous Batching
- Dynamic Batching
- Queue Policy

Memory:

- Paged KV
- Prefix Cache
- Chunked Prefill

Compute:

- CUDA Graph
- FlashAttention
- FlashInfer

Quantization:

- FP8
- AWQ
- SmoothQuant
- GPTQ

## 13. Optimization Roadmap

Không commit cứng. Có thể đổi sau baseline.

```text
Stage 1 -> Stage 2 -> Stage 3
```

## 14. Experiment Log

Format:

```text
Ngày -> Làm gì -> Metric -> Trade-off -> Kết luận
```

## 15. Lessons Learned

Ghi lại sau mỗi experiment có bằng chứng.

## 16. Final Architecture

Architecture cuối, diagram, workflow.

## 17. Final Benchmark

Toàn bộ benchmark cuối.

## 18. Competition Submission

Checklist nộp bài.

## 19. Future Work

Nếu tiếp tục research sau cuộc thi.

## 20. References

- Paper
- Blog
- GitHub
- Documentation

## Decision Log

### Decision 001

Chưa khóa runtime.

Lý do: chưa có model, workload và Docker Compose chính thức từ BTC.

### Decision 002

Dùng Python cho benchmark orchestration.

Lý do: đo đạc, tạo workload, gọi OpenAI-compatible API, tổng hợp metric và tạo report nhanh hơn so với C++/Rust ở giai đoạn tiền đề bài.

### Decision 003

Chưa tải model.

Lý do: tránh lệch khỏi model BTC và tránh khóa sớm vào giả định sai.

## Open Questions

- BTC dùng model gì?
- Có bao nhiêu GPU?
- Có MIG không?
- Có NVLink không?
- Có giới hạn CPU không?
- Có giới hạn RAM/container không?
- Có giới hạn network không?
- Có được dùng image ngoài không?
- Có bắt buộc base image không?
- Chấm quality bằng metric nào?

## Idea Parking Lot

- Adaptive Scheduler
- Online KV Defragmentation
- Dynamic Quantization
- Prefix Prediction
- Request-aware batching
- Quality-aware early stop
- Prefix cache admission policy
