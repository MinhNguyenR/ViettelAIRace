FROM localhost/part2-lfm25-r002k:fused-bf16-greedy

LABEL org.opencontainers.image.title="part2-lfm25-r002aa-shortconv-sparse"
LABEL org.opencontainers.image.description="R002k plus sparse LFM2 ShortConv boundary-state checkpointing"
LABEL org.opencontainers.image.r002aa.padding="0"

COPY local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002aa_lfm2_shortconv_state_checkpoint/overlay/lfm2_all_shortconv.py /opt/r002aa/
COPY local_workspaces/rtx5080/research_rounds/part2_lfm25/round_002_portal_calibrated_decode_first/subrounds/subround_002aa_lfm2_shortconv_state_checkpoint/overlay/sitecustomize.py /opt/r002aa/
ENV PYTHONPATH=/opt/r002aa:/opt/r002k
ENV R002AA_SPARSE_DUAL=1
ENV R002AA_PREFILL_PAD_MULTIPLE=0
