# Cyst Data Analysis

Run from the project root:

```bash
python analysis_data/analyze_dataset.py
```

By default this analyzes `data/train_new.txt`, `data/val_new.txt`, and
`data/test.txt`, matching the non-k-fold training config. To inspect the older
`train.txt`/`val.txt` split, pass `--splits train val test`.

Useful options:

```bash
python analysis_data/analyze_dataset.py --slice-axis 2 --seed 42
python analysis_data/analyze_dataset.py --visuals-per-source 2
python analysis_data/analyze_dataset.py --visual-selection fixed --num-visuals 10
python analysis_data/analyze_dataset.py --skip-visualization
python analysis_data/analyze_dataset.py --output-root outputs_analysis
```
