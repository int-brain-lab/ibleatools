# Cazettes OOP Feature Example

The `oop_compute_cazettes_denoised_features.py` example computes channel
features from local SpikeGLX AP and LF files using
`SpikeGLXFileFeatureCalculator`. It computes five snippets, aggregates the
snippet-level results, denoises the aggregated features, and writes one
per-channel Parquet table.

## Environment

The commands below assume the project and virtual environment are located at:

```text
/home/pranavrai/Work/int-brain-lab/projects/cazettes_sample_data_check
```

Run the example with the Python interpreter from that virtual environment:

```bash
PROJECT_ROOT=/home/pranavrai/Work/int-brain-lab/projects/cazettes_sample_data_check
PYTHON="$PROJECT_ROOT/.venv/bin/python"
SCRIPT="$PROJECT_ROOT/ibleatools/examples/cazettes_lab/oop_compute_cazettes_denoised_features.py"
```

## Recording Layout

The Cazettes recordings use this directory structure:

```text
<data-root>/<subject>/<session-date>/
|-- Rec/
|   `-- <probe-name>/
|       |-- <recording>.ap.cbin
|       |-- <recording>.ap.ch
|       |-- <recording>.ap.meta
|       |-- <recording>.lf.cbin
|       |-- <recording>.lf.ch
|       `-- <recording>.lf.meta
`-- alf/
    `-- <probe-name>/
        `-- channel_locations.json
```

For a general recording:

- `--ap-file` is the file under `Rec/<probe-name>/` ending in `.ap.cbin`.
- `--lf-file` is the matching file ending in `.lf.cbin`.
- `--alf-probe-path` is the matching `alf/<probe-name>/` directory.
- `--output-dir` is the root directory for generated feature files.
- `--name` is a unique recording identifier used for the output folder and
  `pid` column.

Keep each `.cbin` file beside its matching `.ch` and `.meta` files. The ALF
argument is optional; without it, feature computation and denoising still run,
but channel coordinates and atlas labels are not appended to the final table.

Use an explicit `--name` when processing multiple recordings. Filenames such as
`disabled_g0_t0.imec0.ap.cbin` can occur in more than one session, so deriving
the name from the filename alone may not be unique.

## General Command

```bash
DATA_ROOT=/mnt/s0/Data/2026_cazettes/2026_cazettes/Data
OUTPUT_ROOT="$PROJECT_ROOT/ibleatools/examples/cazettes_lab/output/cazettes_oop_denoised_features"

"$PYTHON" "$SCRIPT" \
    --ap-file "$DATA_ROOT/<subject>/<session-date>/Rec/<probe-name>/<recording>.ap.cbin" \
    --lf-file "$DATA_ROOT/<subject>/<session-date>/Rec/<probe-name>/<recording>.lf.cbin" \
    --alf-probe-path "$DATA_ROOT/<subject>/<session-date>/alf/<probe-name>" \
    --output-dir "$OUTPUT_ROOT" \
    --name <subject>_<session-date>_<probe-name>
```

The defaults compute all four feature families (`lf`, `csd`, `ap`, and
`waveforms`) from five 5-second snippets starting at:

```text
300, 600, 900, 1200, and 1500 seconds
```

## Current Recordings

There are three probe recordings under the current data root.

### VF065, 2025_12_17, probe00

```bash
"$PYTHON" "$SCRIPT" \
    --ap-file "$DATA_ROOT/VF065/2025_12_17/Rec/probe00/disabled_g0_t0.imec0.ap.cbin" \
    --lf-file "$DATA_ROOT/VF065/2025_12_17/Rec/probe00/disabled_g0_t0.imec0.lf.cbin" \
    --alf-probe-path "$DATA_ROOT/VF065/2025_12_17/alf/probe00" \
    --output-dir "$OUTPUT_ROOT" \
    --name VF065_2025_12_17_probe00
```

### VF066, 2025_12_04, probe00

```bash
"$PYTHON" "$SCRIPT" \
    --ap-file "$DATA_ROOT/VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.ap.cbin" \
    --lf-file "$DATA_ROOT/VF066/2025_12_04/Rec/probe00/disabled_g0_t0.imec0.lf.cbin" \
    --alf-probe-path "$DATA_ROOT/VF066/2025_12_04/alf/probe00" \
    --output-dir "$OUTPUT_ROOT" \
    --name VF066_2025_12_04_probe00
```

### VF066, 2025_12_04, probe01

```bash
"$PYTHON" "$SCRIPT" \
    --ap-file "$DATA_ROOT/VF066/2025_12_04/Rec/probe01/disabled_g0_t0.imec1.ap.cbin" \
    --lf-file "$DATA_ROOT/VF066/2025_12_04/Rec/probe01/disabled_g0_t0.imec1.lf.cbin" \
    --alf-probe-path "$DATA_ROOT/VF066/2025_12_04/alf/probe01" \
    --output-dir "$OUTPUT_ROOT" \
    --name VF066_2025_12_04_probe01
```

## Custom Snippets

Use `--t-starts`, `--duration-ap`, and `--duration-lf` to override the default
five snippets. For example, this runs one short snippet for an initial check:

```bash
"$PYTHON" "$SCRIPT" \
    --ap-file "$DATA_ROOT/VF065/2025_12_17/Rec/probe00/disabled_g0_t0.imec0.ap.cbin" \
    --lf-file "$DATA_ROOT/VF065/2025_12_17/Rec/probe00/disabled_g0_t0.imec0.lf.cbin" \
    --alf-probe-path "$DATA_ROOT/VF065/2025_12_17/alf/probe00" \
    --output-dir "$OUTPUT_ROOT" \
    --name VF065_2025_12_17_probe00_smoke_test \
    --t-starts 300 \
    --duration-ap 1 \
    --duration-lf 1
```

Use `--skip-saved-computation` to reuse snippet feature files already present
under the selected output directory.

## Outputs

For `--name VF065_2025_12_17_probe00`, outputs are organized as:

```text
<output-dir>/VF065_2025_12_17_probe00/
|-- VF065_2025_12_17_probe00_denoised_features.pqt
|-- VF065_2025_12_17_probe00_snippet_manifest.pqt
|-- oop_feature_cache/
`-- oop_scratch/
```

The `*_denoised_features.pqt` file is the final per-channel feature table. The
manifest records the five snippet directories used during aggregation.
