#!/usr/bin/env bash
# Source after selecting platform/H3_RUN_DIR and before launching any Python process.
case "${H3_DETERMINISTIC:-0}" in
    0) ;;
    1)
        export H3_DETERMINISTIC=1
        export H3_DETERMINISTIC_DEVICE="${platform:?Select the H3 platform first}"
        export H3_DETERMINISM_DIR="${H3_RUN_DIR:?Set the H3 run directory}/determinism"
        export PYTHONHASHSEED=42 VERL_SEED=42
        export VERL_FULL_DETERMINISM=1 VLLM_BATCH_INVARIANT=1
        export CUBLAS_WORKSPACE_CONFIG=:16:8 FLASH_ATTENTION_DETERMINISTIC=1
        export NCCL_DETERMINISTIC=1 NCCL_ALGO=Ring NCCL_PROTO=Simple
        export VERL_DISABLE_FLASH_ATTN_CE=1
        case "$platform" in
            cuda) ;;
            mlu)
                export CNCL_TUNER_PLUGIN="${H3_CNCL_TUNER_DIR:?Set the frozen CNCL tuner directory}/libcncl-tuner-csv.so"
                export CNCL_TUNER_CONFIG_FILE="$H3_CNCL_TUNER_DIR/ring.conf"
                ;;
            *) echo "H3 strict accuracy requires cuda or mlu" >&2; exit 2 ;;
        esac
        ;;
    *) echo "H3_DETERMINISTIC must be 0 or 1" >&2; exit 2 ;;
esac
