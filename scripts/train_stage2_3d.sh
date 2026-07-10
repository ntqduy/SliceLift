#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
clear_config_env_overrides_unless_allowed

# Direct runs use config/stage2_3D as the source of truth.
# Grid scripts enable env overrides internally for generated configs.

if [[ "${ALLOW_ENV_OVERRIDES:-0}" != "1" ]]; then
  run_direct_config "config/stage2_3D/model_experiment_stage2_3d.yaml" "$@"
  exit 0
fi

DECODER_STYLE_TAG="$(decoder_style_value)"
TAG="stage2_3d_$(tag_or_base "${ENC3D:-}")_$(tag_or_base "${DECODER_MODEL:-}")_$(tag_or_base "$DECODER_STYLE_TAG")_seed$(tag_or_base "${SEED:-}")"
begin_generated_config "$TAG" "config/stage2_3D/model_experiment_stage2_3d.yaml"
append_common_run_overrides "$CONFIG_FILE"
append_stage2_model_overrides "$CONFIG_FILE"
append_kfold_override "$CONFIG_FILE"
run_generated_config "$@"
