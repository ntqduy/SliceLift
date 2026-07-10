#!/usr/bin/env bash

# config/*.yaml and config/stage*/ are the source of truth for single-run setup.
# This file only keeps shared script helpers and experiment-grid lists.

PROPOSAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROPOSAL_SCRIPT_DIR/.." && pwd)"
CONFIG_DIR=""
GENERATED_CONFIG_TMP_DIR=""
cd "$ROOT_DIR"

# Optional grid/ablation wrappers can reuse these defaults.
GRID_SEEDS="${GRID_SEEDS:-42 123 205}"
GRID_DECODERS="${GRID_DECODERS:-same_scale nested_dense full_scale}"
GRID_PAIRS="${GRID_PAIRS:-unet3d:unet unet3d:nnunet nnunet3d:unet nnunet3d:nnunet}"
ABLATION_SLICES="${ABLATION_SLICES:-group-based middle uniform random}"

COMMON_PASS_ENV=(OUT_ROOT FOLDS SLICE FUSION NUM_SLICES NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER DECODER_MODEL PARALLEL_FOLDS MAX_PARALLEL_FOLDS PARALLEL_GPU_IDS NUM_WORKERS PERSISTENT_WORKERS PREFETCH_FACTOR)
ABLATION_PASS_ENV=(ENC3D ENC2D DECODER DECODER_MODEL DECODER_STYLE FUSION OUT_ROOT FOLDS NUM_SLICES NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER PARALLEL_FOLDS MAX_PARALLEL_FOLDS PARALLEL_GPU_IDS NUM_WORKERS PERSISTENT_WORKERS PREFETCH_FACTOR)
CONFIG_OVERRIDE_ENV=(SEED OUT_ROOT FOLDS SLICE FUSION NUM_SLICES NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER ENC2D ENC3D DECODER DECODER_MODEL DECODER_STYLE PARALLEL_FOLDS MAX_PARALLEL_FOLDS PARALLEL_GPU_IDS NUM_WORKERS PERSISTENT_WORKERS PREFETCH_FACTOR)

clear_config_env_overrides_unless_allowed() {
  [[ "${ALLOW_ENV_OVERRIDES:-0}" == "1" ]] && return 0
  local name
  for name in "${CONFIG_OVERRIDE_ENV[@]}"; do
    unset "$name" || true
  done
}

tag_or_base() {
  if [[ -n "${1:-}" ]]; then
    printf '%s' "$1"
  else
    printf 'base'
  fi
}

has_any() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] && return 0
  done
  return 1
}

add_env_if_set() {
  local array_name="$1"
  shift
  local -n env_array="$array_name"
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] && env_array+=("$name=${!name}")
  done
  return 0
}

run_script_with_env() {
  local script="$1"
  shift
  env ALLOW_ENV_OVERRIDES=1 "$@" bash "$script" "${ARGS[@]}"
}

cleanup_generated_config_dir() {
  if [[ -n "${GENERATED_CONFIG_TMP_DIR:-}" && -d "$GENERATED_CONFIG_TMP_DIR" && "${KEEP_GENERATED_CONFIGS:-0}" != "1" ]]; then
    rm -rf "$GENERATED_CONFIG_TMP_DIR"
  fi
}

path_for_yaml() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$path"
  else
    printf '%s' "$path"
  fi
}

ensure_generated_config_dir() {
  if [[ -n "${GENERATED_CONFIG_DIR:-}" ]]; then
    CONFIG_DIR="$GENERATED_CONFIG_DIR"
    mkdir -p "$CONFIG_DIR"
    return 0
  fi

  if [[ -z "${GENERATED_CONFIG_TMP_DIR:-}" ]]; then
    GENERATED_CONFIG_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cyst_proposal_config.XXXXXX")"
    trap cleanup_generated_config_dir EXIT
  fi
  CONFIG_DIR="$GENERATED_CONFIG_TMP_DIR"
}

begin_generated_config() {
  local tag="$1"
  local base_path="$2"
  ensure_generated_config_dir
  CONFIG_FILE="$CONFIG_DIR/$tag.yaml"
  BASE_CONFIG="$(path_for_yaml "$ROOT_DIR/$base_path")"
  cat > "$CONFIG_FILE" <<YAML
extends: "$BASE_CONFIG"

YAML
}

append_seed_override() {
  local file="$1"
  [[ -z "${SEED:-}" ]] && return
  cat >> "$file" <<YAML
seed: $SEED
YAML
}

append_output_root_override() {
  local file="$1"
  [[ -z "${OUT_ROOT:-}" ]] && return
  cat >> "$file" <<YAML
paths:
  output_root: $OUT_ROOT
YAML
}

append_common_run_overrides() {
  local file="$1"
  append_seed_override "$file"
  append_output_root_override "$file"
  append_training_loader_override "$file"
}

append_kfold_override() {
  local file="$1"
  has_any FOLDS SEED PARALLEL_FOLDS MAX_PARALLEL_FOLDS PARALLEL_GPU_IDS || return 0
  cat >> "$file" <<YAML
k_fold:
  enabled: true
YAML
  [[ -n "${FOLDS:-}" ]] && printf '  num_folds: %s\n' "$FOLDS" >> "$file"
  [[ -n "${SEED:-}" ]] && printf '  seed: %s\n' "$SEED" >> "$file"
  [[ -n "${PARALLEL_FOLDS:-}" ]] && printf '  run_parallel: %s\n' "$PARALLEL_FOLDS" >> "$file"
  [[ -n "${MAX_PARALLEL_FOLDS:-}" ]] && printf '  max_parallel_folds: %s\n' "$MAX_PARALLEL_FOLDS" >> "$file"
  [[ -n "${PARALLEL_GPU_IDS:-}" ]] && printf '  parallel_gpu_ids: %s\n' "$PARALLEL_GPU_IDS" >> "$file"
  return 0
}

append_training_loader_override() {
  local file="$1"
  has_any NUM_WORKERS PERSISTENT_WORKERS PREFETCH_FACTOR || return 0
  cat >> "$file" <<YAML
training:
YAML
  [[ -n "${NUM_WORKERS:-}" ]] && printf '  num_workers: %s\n' "$NUM_WORKERS" >> "$file"
  [[ -n "${PERSISTENT_WORKERS:-}" ]] && printf '  persistent_workers: %s\n' "$PERSISTENT_WORKERS" >> "$file"
  [[ -n "${PREFETCH_FACTOR:-}" ]] && printf '  prefetch_factor: %s\n' "$PREFETCH_FACTOR" >> "$file"
}

append_type_override() {
  local file="$1"
  local block="$2"
  local var_name="$3"
  local value="${!var_name:-}"
  [[ -z "$value" ]] && return
  cat >> "$file" <<YAML
  $block:
    type: $value
YAML
}

decoder_style_value() {
  printf '%s' "${DECODER_STYLE:-${DECODER:-}}"
}

decoder_model_for_style() {
  local style="${1:-}"
  local encoder_3d="${2:-}"
  if [[ -n "${DECODER_MODEL:-}" ]]; then
    printf '%s' "$DECODER_MODEL"
    return 0
  fi
  case "$style" in
    nested|nested_dense|unetpp|unetplusplus|unet_plus_plus) printf '%s' "unetpp3d" ;;
    full|full_scale|unet3plus|unet_3_plus|full_encoder_single_decoder|all_encoder_single_decoder|full_encoder_one_decoder|all_encoder_one_decoder|single_encoder_full_decoder|single_encoder_full_decode|same_encoder_full_decoder|same_encoder_full_decode|same_scale_full_decoder|same_scale_full_decode|single_skip_full_decoder|single_skip_full_decode) printf '%s' "unet3plus3d" ;;
    same|same_scale|skip|unet)
      case "$encoder_3d" in
        nnunet|nnunet3d|nn_unet|nn_unet3d) printf '%s' "nnunet3d" ;;
        *) printf '%s' "unet3d" ;;
      esac
      ;;
    *) printf '%s' "unet3d" ;;
  esac
}

has_decoder_override() {
  [[ -n "${DECODER_MODEL:-}" || -n "${DECODER_STYLE:-}" || -n "${DECODER:-}" ]]
}

append_decoder_override() {
  local file="$1"
  local block="$2"
  has_decoder_override || return 0
  local style
  style="$(decoder_style_value)"
  echo "  $block:" >> "$file"
  [[ -n "${DECODER_MODEL:-}" ]] && printf '    model: %s\n' "$DECODER_MODEL" >> "$file"
  [[ -n "$style" ]] && printf '    style: %s\n' "$style" >> "$file"
}

append_slice_selection_override() {
  local file="$1"
  has_any SLICE NUM_SLICES SEED NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER || return 0
  echo "  slice_selection:" >> "$file"
  [[ -n "${SLICE:-}" ]] && printf '    mode: %s\n' "$SLICE" >> "$file"
  [[ -n "${NUM_SLICES:-}" ]] && printf '    num_slices: %s\n' "$NUM_SLICES" >> "$file"
  [[ -n "${SEED:-}" ]] && printf '    seed: %s\n' "$SEED" >> "$file"
  if has_any NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER; then
    echo "    group_based:" >> "$file"
    [[ -n "${NUM_GROUPS:-}" ]] && printf '      num_groups: %s\n' "$NUM_GROUPS" >> "$file"
    [[ -n "${SAMPLES_PER_GROUP:-}" ]] && printf '      samples_per_group: %s\n' "$SAMPLES_PER_GROUP" >> "$file"
    [[ -n "${SIMILARITY_METRIC:-}" ]] && printf '      similarity_metric: %s\n' "$SIMILARITY_METRIC" >> "$file"
    [[ -n "${SELECTION_ORDER:-}" ]] && printf '      selection_order: %s\n' "$SELECTION_ORDER" >> "$file"
  fi
  return 0
}

append_fusion_override() {
  local file="$1"
  [[ -z "${FUSION:-}" ]] && return
  cat >> "$file" <<YAML
  fusion:
    type: $FUSION
YAML
}

append_stage1_model_overrides() {
  local file="$1"
  if has_any ENC2D SLICE NUM_SLICES SEED NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER; then
    echo "model:" >> "$file"
    append_type_override "$file" encoder_2d ENC2D
    append_slice_selection_override "$file"
  fi
}

append_stage2_model_overrides() {
  local file="$1"
  if has_any ENC3D DECODER DECODER_MODEL DECODER_STYLE; then
    echo "model:" >> "$file"
    append_type_override "$file" encoder_3d ENC3D
    append_decoder_override "$file" decoder_3d
  fi
}

append_hybrid_model_overrides() {
  local file="$1"
  if has_any ENC2D ENC3D DECODER DECODER_MODEL DECODER_STYLE SLICE NUM_SLICES SEED NUM_GROUPS SAMPLES_PER_GROUP SIMILARITY_METRIC SELECTION_ORDER FUSION; then
    echo "model:" >> "$file"
    append_type_override "$file" encoder_2d ENC2D
    append_type_override "$file" encoder_3d ENC3D
    append_decoder_override "$file" decoder
    append_slice_selection_override "$file"
    append_fusion_override "$file"
  fi
}

run_generated_config() {
  "${PYTHON:-python}" main.py --config "$CONFIG_FILE" "$@"
}

run_direct_config() {
  local config_path="$1"
  shift
  "${PYTHON:-python}" main.py --config "$config_path" "$@"
}
