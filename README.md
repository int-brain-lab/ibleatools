# IBL Electrophysiology Feature Computation and Region Inference

[![Coverage Status](https://coveralls.io/repos/github/int-brain-lab/ibleatools/badge.svg?branch=main)](https://coveralls.io/github/int-brain-lab/ibleatools?branch=main) 
![CI](https://github.com/int-brain-lab/ibleatools/actions/workflows/ci.yml/badge.svg)

This repository contains tools for computing electrophysiology features and performing region inference from neural recordings.

## Documentation

For detailed documentation, installation instructions, usage examples, and API reference, please visit our comprehensive documentation:

**[📚 View Full Documentation](https://int-brain-lab.github.io/ibleatools)**

## Quick Installation

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
> For the **full** install (spike/waveform features via `dartsort`/`dredge`, plus the
> SDSC pipeline), use the `[full]` extra:
>
> ```bash
> pip install -e ".[full]"
> ```

## Contributing

Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to contribute to this project.
