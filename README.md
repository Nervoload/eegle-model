# eegle-model

EEG model workbench for training, inspecting, and comparing EEGNet, standard EEG
baselines, and adapter-based foundation models on public EEG datasets.

The first runnable path is:

- MOABB dataset loading for standard benchmark datasets.
- Braindecode neural baselines: `eegnet`, `shallow`, `deep4`, `eegconformer`
  when your installed Braindecode version exposes them.
- Classic baselines: `riemann_tangent` and `csp_lda`.
- External torch adapters for BENDR, LaBraM, EEGPT, or other cloned
  foundation-model repos.
- JSON run artifacts with resolved config, training history, metrics, and model
  checkpoint.

## WSL/GPU Quickstart

From Ubuntu/WSL in this repo:

```bash
conda activate eegml
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Run a synthetic smoke test first:

```bash
python scripts/train.py --config configs/eegnet_bnci2014_001.yml --smoke
```

Then run EEGNet on MOABB `BNCI2014_001` subject 1. The first run will download
the dataset under `data/moabb`.

```bash
python scripts/train.py --config configs/eegnet_bnci2014_001.yml
```

Compare a shallow ConvNet:

```bash
python scripts/train.py --config configs/shallow_bnci2014_001.yml
```

Compare a Riemannian tangent-space baseline:

```bash
python scripts/train.py --config configs/riemann_bnci2014_001.yml
```

Inspect a model architecture and parameter count:

```bash
python scripts/inspect_model.py --model eegnet --n-chans 22 --n-times 512 --n-outputs 2
```

## ERP/P300 EEGNet Suite

For a Windows/WSL test that is closer to ERP detection or P300 speller work,
start with the P300 suite:

```powershell
.\scripts\windows\run_p300_eegnet.ps1 -CondaEnv eegml -Device cuda
```

Or run the same suite directly from Ubuntu/WSL:

```bash
conda activate eegml
pip install -e .
python scripts/run_suite.py --suite configs/suites/p300_eegnet_windows.yml --device cuda
```

The suite includes:

- `eegnet_p300_synthetic_smoke`: no-download synthetic ERP/P300 check.
- `eegnet_p300_bnci2014_008_subject1`: EEGNet on MOABB `BNCI2014_008`, a
  compact Target-vs-NonTarget P300 dataset.
- `riemann_p300_bnci2014_008_subject1`: a Riemannian baseline for sanity
  checking EEGNet performance.
- `eegnet_p300_epfl_subject6_optional`: disabled by default; enable with
  `--include-disabled` when you want a larger P300 speller-style dataset.

Useful variants:

```bash
python scripts/train.py --config configs/eegnet_p300_bnci2014_008.yml --smoke --device cuda
python scripts/train.py --config configs/eegnet_p300_bnci2014_008.yml --subjects 1 --device cuda
python scripts/run_suite.py --suite configs/suites/p300_eegnet_windows.yml --subjects 1 2 --epochs 10 --device cuda
python scripts/run_suite.py --suite configs/suites/p300_eegnet_windows.yml --include-disabled --device cuda
```

P300/ERP runs write the usual `summary.json`, `metrics.json`, and model
checkpoint, plus `predictions.csv`. The prediction file includes true labels,
predicted labels, class scores, and any MOABB metadata carried through the test
split. That file is the bridge from event-level ERP detection toward
dataset-specific character/speller decoding, where flashes must be grouped back
into a selection sequence.

## Config Shape

Experiment configs live in `configs/` and have three main blocks:

- `dataset`: source, MOABB dataset class, paradigm, subjects, filtering/window
  options, and standardization.
- `model`: model name plus model-specific keyword arguments or external adapter
  details.
- `training`: epochs, batch size, learning rate, split sizes, seed, device, and
  mixed precision.

CLI overrides are intentionally small:

```bash
python scripts/train.py \
  --config configs/eegnet_bnci2014_001.yml \
  --subjects 1 2 3 \
  --epochs 50 \
  --batch-size 128 \
  --device cuda
```

## Foundation Models

BENDR and LaBraM are not treated as ordinary pip-installed model classes here.
They are research repos with their own training scripts, checkpoint formats, and
preprocessing expectations. The workbench therefore provides an external torch
adapter contract in `configs/foundation_external_template.yml`:

```yaml
model:
  name: external_torch
  framework: torch
  external:
    repo_path: /home/surettej/eeg-foundation/BENDR
    module: workbench_adapter
    factory: build_classifier
    checkpoint: /home/surettej/eeg-foundation/checkpoints/bendr.pt
    strict: false
    kwargs: {}
```

Create `workbench_adapter.py` in the cloned foundation repo and expose a
`build_classifier(...)` function. The factory can accept any of these shape
arguments if useful: `n_chans`, `in_chans`, `n_outputs`, `n_classes`,
`num_classes`, `n_times`, or `input_window_samples`.

The returned module must:

- accept `float32` tensors shaped `(batch, channels, time)`;
- return logits shaped `(batch, classes)`;
- own any foundation-specific preprocessing, tokenization, or classification
  head decisions.

That keeps the benchmarking pipeline stable while letting each foundation model
keep its native implementation.

## Outputs

Each run writes a timestamped folder under `runs/`:

- `config.resolved.yml`
- `metrics.json`
- `summary.json`
- `model.pt` for torch models or `model.joblib` for sklearn models

The default metrics are accuracy, balanced accuracy, macro F1, confusion matrix,
and a full sklearn classification report.

## Notes

- `data/`, `runs/`, and checkpoints are ignored by Git.
- For fair comparisons, keep dataset, subject split, filtering, resampling, and
  standardization fixed while changing only the model config.
- Foundation-model comparisons need extra care because channel layouts,
  sampling rates, and pretraining tokenizers often differ from MOABB trial
  tensors.

## References

- EEGNet paper: https://arxiv.org/abs/1611.08024
- Braindecode project/models: https://github.com/braindecode/braindecode
- MOABB documentation: https://moabb.neurotechx.com/docs/
- MOABB P300 paradigm: https://moabb.neurotechx.com/docs/generated/moabb.paradigms.P300.html
- MOABB BNCI2014_008 P300 dataset: https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html
- MOABB EPFLP300 dataset: https://moabb.neurotechx.com/docs/generated/moabb.datasets.EPFLP300.html
- BENDR repository: https://github.com/SPOClab-ca/BENDR
- LaBraM repository: https://github.com/935963004/LaBraM
