# Cazettes OOP Feature Example

This folder contains one example for the `VF066`, `2025_12_04`, `probe00`
recording. It computes OOP channel features from five snippets, aggregates them,
and saves the denoised feature table.

## Run

From the project root:

```bash
.venv/bin/python \
    ibleatools/examples/cazettes_lab/oop_compute_cazettes_denoised_features.py
```

The recording paths and feature settings are constants near the top of the
script. No command-line arguments are required.

## Example Recording

The script reads:

```text
/mnt/s0/Data/2026_cazettes/2026_cazettes/Data/VF066/2025_12_04/Rec/probe00/
|-- disabled_g0_t0.imec0.ap.cbin
|-- disabled_g0_t0.imec0.ap.ch
|-- disabled_g0_t0.imec0.ap.meta
|-- disabled_g0_t0.imec0.lf.cbin
|-- disabled_g0_t0.imec0.lf.ch
`-- disabled_g0_t0.imec0.lf.meta
```

The `.ch` and `.meta` companion files must remain beside their corresponding
`.cbin` files.

By default, the script computes `lf`, `csd`, `ap`, and `waveforms` features from
five 5-second snippets starting at:

```text
300, 600, 900, 1200, and 1500 seconds
```

## Use Another Recording

Edit these constants in the script:

```python
RECORDING_NAME = "subject_session_probe"
AP_FILE = DATA_ROOT / "subject/session/Rec/probe/recording.ap.cbin"
LF_FILE = DATA_ROOT / "subject/session/Rec/probe/recording.lf.cbin"
```

The expected directory pattern is:

```text
<data-root>/<subject>/<session>/Rec/<probe>/<recording>.ap.cbin
<data-root>/<subject>/<session>/Rec/<probe>/<recording>.lf.cbin
```

`SNIPPET_T_STARTS`, `DURATION_AP`, `DURATION_LF`, and `FEATURES_TO_COMPUTE` can
also be edited near the top of the script.

## Outputs

Outputs are written below:

```text
examples/cazettes_lab/output/VF066_2025_12_04_probe00/
|-- VF066_2025_12_04_probe00_denoised_features.pqt
|-- VF066_2025_12_04_probe00_snippet_manifest.pqt
|-- feature_cache/
`-- scratch/
```
