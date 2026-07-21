#!/usr/bin/env bash

r002aa_require_idle_gpu() {
  if (( $# != 1 )); then
    echo "usage: r002aa_require_idle_gpu OUTPUT_DIR" >&2
    return 2
  fi

  local output_dir="$1"
  local require_idle="${R002AA_REQUIRE_IDLE_GPU:-1}"
  local max_util="${R002AA_IDLE_MAX_GPU_UTIL:-10}"
  local hard_max_util="${R002AA_IDLE_HARD_MAX_GPU_UTIL:-20}"
  local max_util_spikes="${R002AA_IDLE_MAX_UTIL_SPIKES:-1}"
  local max_memory="${R002AA_IDLE_MAX_MEMORY_MIB:-2048}"
  local sample_count="${R002AA_IDLE_SAMPLE_COUNT:-5}"
  local settle_seconds="${R002AA_IDLE_SETTLE_SECONDS:-0}"
  local sample_file="$output_dir/gpu_idle_samples.csv"
  local sample

  if [[ "$require_idle" != "0" && "$require_idle" != "1" ]]; then
    echo "R002AA_REQUIRE_IDLE_GPU must be 0 or 1" >&2
    return 2
  fi
  for value in \
    "$max_util" \
    "$hard_max_util" \
    "$max_util_spikes" \
    "$max_memory" \
    "$sample_count" \
    "$settle_seconds"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
      echo "R002aa idle-GPU thresholds must be non-negative integers" >&2
      return 2
    fi
  done
  if (( sample_count < 1 )); then
    echo "R002AA_IDLE_SAMPLE_COUNT must be positive" >&2
    return 2
  fi

  if (( settle_seconds > 0 )); then
    sleep "$settle_seconds"
  fi

  printf 'timestamp,memory_total_mib,memory_used_mib,memory_free_mib,gpu_util_percent,power_w\n' \
    >"$sample_file"
  for (( sample = 0; sample < sample_count; sample++ )); do
    printf '%s,' "$(date --iso-8601=seconds)" >>"$sample_file"
    nvidia-smi \
      --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,power.draw \
      --format=csv,noheader,nounits >>"$sample_file"
    if (( sample + 1 < sample_count )); then
      sleep 1
    fi
  done

  if [[ "$require_idle" == "1" ]] && ! awk -F, \
    -v max_util="$max_util" \
    -v hard_max_util="$hard_max_util" \
    -v max_util_spikes="$max_util_spikes" \
    -v max_memory="$max_memory" '
      NR == 1 { next }
      {
        gsub(/[[:space:]]/, "", $3)
        gsub(/[[:space:]]/, "", $5)
        if (($3 + 0) > max_memory || ($5 + 0) > hard_max_util) {
          rejected = 1
        }
        if (($5 + 0) > max_util) {
          util_spikes += 1
        }
      }
      END { exit (rejected || util_spikes > max_util_spikes) ? 1 : 0 }
    ' "$sample_file"; then
    echo "R002aa idle-GPU gate rejected the run; inspect $sample_file" >&2
    return 3
  fi
}
