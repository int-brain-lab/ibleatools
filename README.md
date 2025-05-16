# IBL Electrophysiology Feature Computation and Region Inference

This repository contains tools for computing electrophysiology features and performing region inference from neural recordings.

## Installation

1. First, activate the IBL conda environment:
```bash
conda activate iblenv
```

2. Install the required packages using conda:
```bash
conda install --file requirements.txt
```

If some packages are not available through conda, you can install them using pip:
```bash
pip install -r requirements.txt
```

## Usage

The main interface is through `main.py`, which can be run using a configuration file:

```bash
python main.py --config config.yaml
```

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
features_path: "/path/to/features"  # Path to save/load features file
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
  - `features_path`: Path to save/load features file. If not provided, features will be saved as `features_{pid}.parquet` in the current directory
  - `model_path`: Path to the model directory for region inference. If not provided, a default path will be used

## Features

The tool performs two main operations:

1. **Feature Computation**:
   - Computes various electrophysiology features from the raw data
   - Saves features in Parquet format for efficient storage and retrieval

2. **Region Inference**:
   - Uses pre-trained models to infer brain regions
   - Can be run independently if features are already computed

## Output

- Features are saved in Parquet format for efficient storage
- Region inference results include predicted regions and their probabilities
- All operations are logged for tracking and debugging
