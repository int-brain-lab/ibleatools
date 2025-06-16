# IBL Electrophysiology Feature Computation and Region Inference

This repository contains tools for computing electrophysiology features and performing region inference from neural recordings.

## Installation

> **Note**: It is recommended to create and use a separate virtual environment before installation.

1. Clone the repository and navigate to the directory:
```bash
git clone https://github.com/int-brain-lab/ibleatools.git
cd ibleatools
```

2. Install the package in editable mode:
```bash
pip install -e .
```

## Main Functions

The package provides two main functions for electrophysiology analysis:

### 1. Feature Computation (`compute_features`)

This function computes various electrophysiological features from raw neural recordings. It can work with either:
- Data from the IBL database using a probe ID (pid)
- Local .cbin files (AP and LF band data)

Basic usage:
```python
from one.api import ONE
from ephysatlas.feature_computation import compute_features

# Using IBL database
one = ONE()  # Initialize ONE client
df_features = compute_features(
    pid="your_probe_id",
    t_start=300.0,  # Start time in seconds
    duration=3.0,   # Duration in seconds
    one=one
)

# Using local files
df_features = compute_features(
    ap_file="path/to/ap.cbin",
    lf_file="path/to/lf.cbin",
    t_start=300.0,
    duration=3.0
)
```

The function returns a pandas DataFrame containing various electrophysiological features, which are also saved in Parquet format for efficient storage and retrieval.

> **Note**: Due to a known issue in PyTorch ([#132372](https://github.com/pytorch/pytorch/issues/132372)), you might encounter a SEGFAULT when running the feature computation. To resolve this, you can either:
> 1. Import torch at the start of your script:
>    ```python
>    import torch  # Add this at the beginning of your script
>    ```
> 2. Set the `DYLD_LIBRARY_PATH` environment variable to point to your virtual environment's torch library:
>    ```bash
>    export DYLD_LIBRARY_PATH=/path/to/your/venv/lib/python3.x/site-packages/torch/lib
>    ```

> **Important**: This package (`ephysatlas`) is different from the `ephys_atlas` package (with underscore) from the [paper-ephys-atlas](https://github.com/int-brain-lab/paper-ephys-atlas) repository.

### 2. Region Inference (`infer_regions`)

This function uses pre-trained models to infer brain regions from the computed features. It performs inference across multiple model folds and returns both the predicted regions and their probabilities.

Basic usage:
```python
from ephysatlas.region_inference import infer_regions

# Perform region inference
predicted_probas, predicted_region = infer_regions(
    df_inference=df_features,  # DataFrame from compute_features
    path_model="path/to/model"  # Path to the model directory
)
```

The function returns:
- `predicted_probas`: Array of shape (n_folds, n_channels, n_regions) containing region probabilities
- `predicted_region`: Array of shape (n_folds, n_channels) containing predicted region indices

## Usage through CLI

The CLI interface is through `main.py`, which can be run using a configuration file:

```bash
python main.py --config config.yaml
```

Using CLI one can do both feature computations and region inference by specifying it in the configuration

### Configuration File

The configuration is managed through a YAML file. To avoid committing local changes, the actual configuration file (`config.yaml`) is ignored by git. Instead, a template file (`config_template.yaml`) is provided. To use the tool:

1. Copy the template file to create your local configuration:
```bash
cp config_template.yaml config.yaml
```

2. Edit `config.yaml` with your specific settings:
```yaml
# Required parameters
pid: "5246af08-0730-40f7-83de-29b5d62b9b6d"  # Probe ID
t_start: 300.0  # Start time in seconds
duration: 3.0  # Duration in seconds

# Operation mode
mode: "both"  # Options: 'features', 'inference', or 'both'

# Optional parameters
output_dir: "/path/to/output_dir"  # Path to output directory
model_path: "/path/to/model"  # Path to the model directory for region inference
```

#### Configuration Parameters

- **Required Parameters**:
  - `pid`: Probe ID for the recording
  - `t_start`: Start time in seconds
  - `duration`: Duration of the analysis in seconds

- **Operation Mode**:
  - `mode`: Specifies which operations to perform
    - `features`: Only compute features
    - `inference`: Only perform region inference
    - `both`: Perform both feature computation and region inference

- **Optional Parameters**:
  - `output_dir`: Path to output directory for saving results
  - `model_path`: Path to the model directory for region inference. If not provided, a default path will be used

## Output

- Features are saved in Parquet format for efficient storage
- Region inference results include predicted regions and their probabilities