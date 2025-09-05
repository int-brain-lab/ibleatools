.. ibleatools documentation master file, created by
   sphinx-quickstart on Mon Aug 25 02:02:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ibleatools Documentation
========================

Welcome to the ibleatools documentation! This package provides comprehensive tools for electrophysiological data analysis, including feature extraction, visualization, and brain region classification.

What is ibleatools?
--------------------

ibleatools is a Python package designed for the analysis of electrophysiological data from the International Brain Laboratory (IBL). It provides:

* **Feature Extraction**: Comprehensive extraction of electrophysiological features from AP and LF bands
* **Data Visualization**: Plotting tools for probe data and brain regions
* **Brain Region Classification**: Machine learning models for automatic brain region identification
* **Data Management**: Utilities for organizing and managing large-scale electrophysiological datasets



Installation
-------------

.. note::
   It is recommended to create and use a separate virtual environment before installation.

1. Clone the repository and navigate to the directory:

.. code-block:: bash

   git clone https://github.com/int-brain-lab/ibleatools.git
   cd ibleatools

2. Install the package in editable mode:

.. code-block:: bash

   pip install -e .

Main Functions
--------------

The package provides functions for electrophysiology analysis:

1. Feature Computation from Probe ID (`compute_features_from_pid`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function computes various electrophysiological features from raw neural recordings using data from the IBL database with a probe ID (pid).

Basic usage:

.. code-block:: python

   from one.api import ONE
   from ephysatlas.feature_computation import compute_features_from_pid

   # Using IBL database
   one = ONE()  # Initialize ONE client
   df_features = compute_features_from_pid(
       pid="your_probe_id",
       t_start=300.0,  # Start time in seconds
       duration=3.0,   # Duration in seconds
       one=one
   )

The function returns a pandas DataFrame containing various electrophysiological features, which are also saved in Parquet format for efficient storage and retrieval.

2. Feature Computation from Files (`compute_features_from_file`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function computes various electrophysiological features from local .cbin files (AP and LF band data).

Basic usage:

.. code-block:: python

   from ephysatlas.feature_computation import compute_features_from_file

   # Using local files
   df_features = compute_features_from_file(
       ap_file="path/to/ap.cbin",
       lf_file="path/to/lf.cbin",
       t_start=300.0,
       duration=3.0
   )

3. Legacy Function (`compute_features`) - DEPRECATED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   The `compute_features` function is deprecated and will be removed in a future version. Please use `compute_features_from_pid` or `compute_features_from_file` instead.

This function was the original interface for computing electrophysiological features. It can work with either:
- Data from the IBL database using a probe ID (pid)
- Local .cbin files (AP and LF band data)

Basic usage:

.. code-block:: python

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

The function returns a pandas DataFrame containing various electrophysiological features, which are also saved in Parquet format for efficient storage and retrieval.

.. note::
   Due to a known issue in PyTorch (`#132372 <https://github.com/pytorch/pytorch/issues/132372>`_), you might encounter a SEGFAULT when running the feature computation. To resolve this, you can either:

   1. Import torch at the start of your script:

      .. code-block:: python

         import torch  # Add this at the beginning of your script

   2. Set the `DYLD_LIBRARY_PATH` environment variable to point to your virtual environment's torch library:

      .. code-block:: bash

         export DYLD_LIBRARY_PATH=/path/to/your/venv/lib/python3.x/site-packages/torch/lib

.. important::
   This package (`ephysatlas`) is different from the `ephys_atlas` package (with underscore) from the `paper-ephys-atlas <https://github.com/int-brain-lab/paper-ephys-atlas>`_ repository.

4. Region Inference (`infer_regions`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function uses pre-trained models to infer brain regions from the computed features. It performs inference across multiple model folds and returns both the predicted regions and their probabilities.

Basic usage:

.. code-block:: python

   from ephysatlas.regionclassifier.region_inference import infer_regions

   # Perform region inference
   predicted_probas, predicted_region = infer_regions(
       df_inference=df_features,  # DataFrame from compute_features
       path_model="path/to/model"  # Path to the model directory
   )

The function returns:
- `predicted_probas`: Array of shape (n_folds, n_channels, n_regions) containing region probabilities
- `predicted_region`: Array of shape (n_folds, n_channels) containing predicted region indices

Usage through CLI
-----------------

The CLI interface is through `main.py`, which can be run using a configuration file:

.. code-block:: bash

   python main.py --config config.yaml

Using CLI one can do both feature computations and region inference by specifying it in the configuration.

.. note::
   The CLI currently uses the deprecated `compute_features` function internally. This will be updated in a future version to use the new `compute_features_from_pid` and `compute_features_from_file` functions.

Configuration File
~~~~~~~~~~~~~~~~~~

The configuration is managed through a YAML file. To avoid committing local changes, the actual configuration file (`config.yaml`) is ignored by git. Instead, a template file (`config_template.yaml`) is provided. To use the tool:

1. Copy the template file to create your local configuration:

.. code-block:: bash

   cp config_template.yaml config.yaml

2. Edit `config.yaml` with your specific settings:

.. code-block:: yaml

   # Required parameters
   pid: "5246af08-0730-40f7-83de-29b5d62b9b6d"  # Probe ID
   t_start: 300.0  # Start time in seconds
   duration: 3.0  # Duration in seconds

   # Operation mode
   mode: "both"  # Options: 'features', 'inference', or 'both'

   # Optional parameters
   output_dir: "/path/to/output_dir"  # Path to output directory
   model_path: "/path/to/model"  # Path to the model directory for region inference

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

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

Output
------

- Features are saved in Parquet format for efficient storage
- Region inference results include predicted regions and their probabilities

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ephysatlas
   how-to/index
   reference/index
   features
   plots
   reveal
   utils

Getting Help
------------

* **How-to Guides**: Step-by-step tutorials for common tasks
* **API Reference**: Complete documentation of all functions and classes
* **Examples**: Working code examples you can copy and modify
* **Source Code**: Well-documented source code with Google Style docstrings

For questions and support, please check the documentation or open an issue on the project repository.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`