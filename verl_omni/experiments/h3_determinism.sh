#!/usr/bin/env bash
# Source after selecting platform/H3_RUN_DIR and before launching any Python process.
case "${H3_DETERMINISTIC:-0}" in
    0) ;;
    1)
        export H3_DETERMINISTIC=1
        export H3_DETERMINISTIC_DEVICE="${platform:?Select the H3 platform first}"
        export H3_DETERMINISM_DIR="${H3_RUN_DIR:?Set the H3 run directory}/determinism"
        case "$platform" in
            cuda) export CUBLAS_WORKSPACE_CONFIG=:4096:8 ;;
            mlu) ;;
            *) echo "H3 kernel determinism requires cuda or mlu" >&2; exit 2 ;;
        esac
        ;;
    *) echo "H3_DETERMINISTIC must be 0 or 1" >&2; exit 2 ;;
esac
