#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
clear_config_env_overrides_unless_allowed

# Direct runs use config/stage3_hybrid as the source of truth.
# Grid/ablation scripts enable env overrides internally for generated configs.

if [[ "${ALLOW_ENV_OVERRIDES:-0}" != "1" ]]; then
  run_direct_config "config/stage3_hybrid/model_experiment_hybrid.yaml" "$@"
  exit 0
fi

DECODER_STYLE_TAG="$(decoder_style_value)"
TAG="hybrid_$(tag_or_base "${ENC3D:-}")_$(tag_or_base "${ENC2D:-}")_$(tag_or_base "${DECODER_MODEL:-}")_$(tag_or_base "$DECODER_STYLE_TAG")_$(tag_or_base "${FUSION:-}")_$(tag_or_base "${SLICE:-}")_no_pos_seed$(tag_or_base "${SEED:-}")"
begin_generated_config "$TAG" "config/stage3_hybrid/model_experiment_hybrid.yaml"
append_common_run_overrides "$CONFIG_FILE"
append_hybrid_model_overrides "$CONFIG_FILE"
append_kfold_override "$CONFIG_FILE"
run_generated_config "$@"
