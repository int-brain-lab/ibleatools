# IBL Electrophysiology Feature Computation and Region Inference

[![Coverage Status](https://coveralls.io/repos/github/int-brain-lab/ibleatools/badge.svg?branch=main)](https://coveralls.io/github/int-brain-lab/ibleatools?branch=main) 
![CI](https://github.com/int-brain-lab/ibleatools/actions/workflows/ci.yml/badge.svg)

This repository contains tools for computing electrophysiology features and performing region inference from neural recordings.

## Documentation

**[📚 View Full Documentation](https://int-brain-lab.github.io/ibleatools)**

| Start here | |
|---|---|
| [Tutorial](https://int-brain-lab.github.io/ibleatools/tutorials/first-predictions.html) | Install, download a published model, predict brain regions |
| [Compute features on your own data](https://int-brain-lab.github.io/ibleatools/how-to/compute-features-spikeglx.html) | SpikeGLX, NWB/DANDI or SpikeInterface-Zarr recordings |
| [Run a published model](https://int-brain-lab.github.io/ibleatools/how-to/download-models.html) | Region decoder and spatial encoder, from the Hugging Face Hub |
| [Plot features along the probe](https://int-brain-lab.github.io/ibleatools/how-to/plot-features-along-probe.html) | Visualise how features vary with depth |
| [API reference](https://int-brain-lab.github.io/ibleatools/api/) | Generated from the docstrings |

The site is a [Quarto](https://quarto.org/) website built from `docs/`; see
[`docs/README.txt`](docs/README.txt) to build it locally.

## Published models

Both are public Hugging Face repositories — no account or token needed — and load through
the single [`load_pretrained`](https://int-brain-lab.github.io/ibleatools/api/load_pretrained.html) entry point:

- [`int-brain-lab/ea-decoder-channel-xgboost`](https://huggingface.co/int-brain-lab/ea-decoder-channel-xgboost) — brain region per channel, from features
- [`int-brain-lab/ea-encoder-channel`](https://huggingface.co/int-brain-lab/ea-encoder-channel) — features per channel, from position

```python
from ephysatlas import load_pretrained

model = load_pretrained(
    "int-brain-lab/ea-decoder-channel-xgboost", revision="2026_W32"
)
model.selftest()
out = model.predict(df_features)
```

## Quick Installation

ibleatools requires Python 3.12 or newer.

```bash
git clone https://github.com/int-brain-lab/ibleatools.git
cd ibleatools
pip install -e .
```

> **Lite vs full install.** The command above installs the **lite** version, which is
> the default. Lite was developed for
> [`ibl-alignment-gui`](https://github.com/int-brain-lab/ibl-alignment-gui) (which
> depends on it as `ibleatools[lite]`) and covers S3 table/model downloads,
> region-classifier inference, the spatial encoder, and LF/CSD/AP feature computation —
> but **not** spike/waveform feature computation.
>
> For the **full** install (spike/waveform features via `dartsort`, plus the
> SDSC pipeline), use the `[full]` extra:
>
> ```bash
> pip install -e ".[full]"
> ```

## Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to contribute to this project.
