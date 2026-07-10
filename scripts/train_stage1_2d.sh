#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
clear_config_env_overrides_unless_allowed

# Direct runs use config/stage1_2D as the source of truth.
# Grid/ablation scripts enable env overrides internally for generated configs.

if [[ "${ALLOW_ENV_OVERRIDES:-0}" != "1" ]]; then
  run_direct_config "config/stage1_2D/model_experiment_stage1_2d.yaml" "$@"
  exit 0
fi

TAG="stage1_2d_$(tag_or_base "${ENC2D:-}")_$(tag_or_base "${SLICE:-}")_no_pos_seed$(tag_or_base "${SEED:-}")"
begin_generated_config "$TAG" "config/stage1_2D/model_experiment_stage1_2d.yaml"
append_common_run_overrides "$CONFIG_FILE"
append_stage1_model_overrides "$CONFIG_FILE"
append_kfold_override "$CONFIG_FILE"
run_generated_config "$@"
