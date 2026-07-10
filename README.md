# SliceLift: Slice-to-Volume Feature Lifting for Pancreatic Cyst MRI Segmentation

Paper-ready code for cyst segmentation with a staged 2D/3D hybrid architecture. The repository focuses on one proposal family:

1. Stage 1: train a 2D encoder-decoder from selected slices.
2. Stage 2: train a 3D encoder-decoder from full volumes.
3. Stage 3: train a hybrid model that selects informative 2D slices, lifts 2D features into the 3D feature pyramid, fuses them with 3D encoder stages, and decodes the fused 3D representation.


## Main Idea

The hybrid model keeps the proposal structure intact:

- A 2D branch extracts high-resolution slice features.
- A 3D branch preserves volumetric context.
- Slice selection can be `group-based`, `middle`, `uniform`, or `random`.
- 2D features are scattered back to the corresponding 3D depth locations.
- The 3D decoder can use U-Net-style, UNet++-style, UNet3+-style, or ablation decoder styles.

## Repository Structure

```text
SliceVolumeFusion-Cyst/
  main.py                         # CLI entrypoint
  requirements.txt                # Python dependencies
  config/
    model_experiment_base.yaml    # shared experiment config
    stage1_2D/                    # Stage 1 2D pretraining config
    stage2_3D/                    # Stage 2 3D pretraining config
    stage3_hybrid/                # Stage 3 hybrid and ablation configs
  baseline/
    Proposal_Model_Experiment/
      2D_encoder/                 # UNet, UNet++, UNet3+, nnUNet-style 2D encoders
      3D_encoder/                 # UNet3D, UNet++3D, UNet3+3D, nnUNet-style 3D encoders
      Decoder/                    # same-scale, nested-dense, full-scale, ablation decoders
      HybridModel/                # Stage 1, Stage 2, and hybrid model definitions
      selection_slice_2D.py       # slice selection module
  training/
    data.py                       # NIfTI datasets, k-fold splits, slice/volume loading
    model_factory.py              # compact proposal model factory
    runner.py                     # train/evaluate/k-fold/checkpoint/visualization loop
    losses.py, metrics.py         # loss and metric wrappers
    visualization.py              # qualitative outputs and diagnostics
  losses/
    bce.py                        # binary BCEWithLogits loss for 1- or 2-channel logits
    dice.py                       # binary soft Dice helpers/loss
    composite.py                  # ce_dice/dice_bce criterion builder
  scripts/
    train_stage1_2d.sh
    train_stage2_3d.sh
    train_hybrid.sh
    common.sh
  data/                           # put split CSV files and NIfTI data here
  outputs/                        # generated training outputs
```

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, use Git Bash, WSL, or run the Python commands directly instead of the `.sh` scripts.

## Data Layout

By default, configs read from `data/`.

Required split files:

```text
data/train_new.txt
data/val_new.txt
data/all_train.txt
data/test.txt
```

Each file is a CSV with:

```csv
image_path,mask_path
images/case_001.nii.gz,masks/case_001.nii.gz
```

Paths can be absolute, or relative to `data/`. `all_train.txt` is used for k-fold train/validation splitting. `test.txt` is always kept as the held-out test set.

## Configuration

The shared config is:

```text
config/model_experiment_base.yaml
```

Stage-specific entry configs:

```text
config/stage1_2D/model_experiment_stage1_2d.yaml
config/stage2_3D/model_experiment_stage2_3d.yaml
config/stage3_hybrid/model_experiment_hybrid.yaml
```

Important knobs:

```yaml
model:
  encoder_2d:
    type: unet | unetpp | unet3plus | nnunet
  encoder_3d:
    type: unet3d | unetpp3d | unet3plus3d | nnunet3d
  decoder:
    model: unet3d | unetpp3d | unet3plus3d | nnunet3d
    style: same_scale | nested_dense | full_scale | full_encoder_single_decoder | single_encoder_full_decoder
  slice_selection:
    mode: group-based | middle | uniform | random
    group_based:
      num_groups: 15
      samples_per_group: 1
      similarity_metric: mad
      selection_order: different
  fusion:
    type: add | concat

training:
  loss: ce_dice
```

In `group-based` mode, the selected slice count is resolved from `num_groups * samples_per_group`.

### Loss

The default method uses:

```text
ce_dice -> dice_bce = 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss
```

The kept loss modules are intentionally minimal:

```text
losses/dice.py
losses/bce.py
losses/composite.py
```

## Training

Run the three stages in order when Stage 3 should load pretrained Stage 1/2 components.

```bash
bash scripts/train_stage1_2d.sh
bash scripts/train_stage2_3d.sh
bash scripts/train_hybrid.sh
```

Equivalent direct Python commands:

```bash
python main.py --config config/stage1_2D/model_experiment_stage1_2d.yaml
python main.py --config config/stage2_3D/model_experiment_stage2_3d.yaml
python main.py --config config/stage3_hybrid/model_experiment_hybrid.yaml
```

Stage 3 expects Stage 1 and Stage 2 checkpoints unless disabled in config:

```yaml
pretrain:
  load_2d_encoder_from_stage1: true
  load_3d_encoder_from_stage2: true
  load_3d_decoder_from_stage2: true
```

To run hybrid ablations without Stage 2 loading:

```bash
python main.py --config config/stage3_hybrid/model_experiment_hybrid_no_stage2_full_encoder_single_decoder.yaml
python main.py --config config/stage3_hybrid/model_experiment_hybrid_no_stage2_single_encoder_full_decoder.yaml
```

## K-Fold

K-fold is enabled in `config/model_experiment_base.yaml`:

```yaml
k_fold:
  enabled: true
  source_list: all_train.txt
  num_folds: 3
  shuffle: true
  seed: 42
```

Splits are saved and reused under:

```text
data/data_fold/<source>__<test>__k<num_folds>__seed<seed>__<shuffle>/
```

Final k-fold summaries are written to:

```text
metrics_kfold.csv
metrics_kfold_summary.csv
kfold_splits.json
```

## Evaluation and Visualization

To regenerate visualization from a checkpoint:

```bash
python main.py \
  --config config/stage3_hybrid/model_experiment_hybrid.yaml \
  --visualize-only \
  --checkpoint outputs/Proposal_Model_Experiment/3_hybrid/.../checkpoints/best_model.pth \
  --output-dir outputs/visualize_hybrid
```

Use `--use-checkpoint-config` when the checkpoint contains the exact config to reuse:

```bash
python main.py --visualize-only --use-checkpoint-config --checkpoint path/to/best_model.pth --output-dir outputs/visualize_case
```


## Citation

If this code is used in a publication, cite the associated paper and include the exact config/checkpoint metadata generated by this repository.
