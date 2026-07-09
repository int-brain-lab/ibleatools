API Reference
=============

This section provides comprehensive API documentation for all modules, classes, and functions in ibleatools.

Package Overview
-----------------

ibleatools is organized into several core modules, each providing specific functionality for electrophysiological data analysis:

* **ephysatlas**: Main package containing all modules
* **features**: Feature extraction and computation
* **plots**: Visualization and plotting utilities
* **reveal**: High-level analysis and figure generation
* **utils**: Utility functions and data management
* **anatomy**: Brain region classification
* **regionclassifier**: Machine learning models
* **data**: Data loading and management

Module Documentation
--------------------------------------

Core Modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   ../ephysatlas
   ../features
   ../plots
   ../reveal
   ../utils

.. Specialized Modules
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. toctree::
..    :maxdepth: 2

..    anatomy
..    regionclassifier
..    data
..    feature_computation

.. Configuration
.. -------------

.. .. toctree::
..    :maxdepth: 2

..    configuration
..    schemas
..    parameters

.. Data Structures
.. ----------------------------

.. .. toctree::
..    :maxdepth: 2
   
..    dataframes
..    models
..    transformers

Quick Reference
----------------------------

Common Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Feature Extraction**: :func:`ephysatlas.feature_computation.compute_features_from_pid` / :func:`~ephysatlas.feature_computation.compute_features_from_file`
* **OOP calculators**: :class:`ephysatlas.feature_calculators.IBLPIDFeatureCalculator`, :class:`~ephysatlas.feature_calculators.SpikeGLXFileFeatureCalculator`
* **Visualization**: :class:`ephysatlas.reveal.AtlasReveal`
* **Data Loading**: :func:`ephysatlas.data.load_raw_data`
* **Utilities**: :func:`ephysatlas.utils.setup_output_directory`

Data Types
~~~~~~~~~~~~~~~~~~~~

* **Features**: :class:`ephysatlas.features.ModelRawFeatures`
* **Parameters**: :class:`ephysatlas.features.DartParameters`
* **Transformers**: :class:`ephysatlas.features.EphysTransformer`

Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

* **Schemas**: Pandera-based data validation schemas
* **Parameters**: Configurable parameters for all major functions
* **Defaults**: Sensible defaults for common use cases

Usage Patterns
--------------------------

**Basic Workflow:**

1. Load data using a PID
2. Extract features using appropriate functions
3. Visualize results with plotting functions
4. Save outputs for further analysis

**Advanced Workflow:**

1. Customize parameters for specific needs
2. Use transformers for data preprocessing
3. Generate comprehensive figures with AtlasReveal
4. Apply machine learning models for classification

**Data Management:**

1. Set up output directories with utilities
2. Manage metadata and file organization
3. Aggregate results across multiple experiments
4. Export data in various formats

Examples
--------

For practical examples and step-by-step guides, see the :doc:`../how-to/index` section.

Getting Help
----------------------

* **Documentation**: This reference guide
* **Examples**: Code examples in the :doc:`../how-to/index` section
* **Source Code**: Full source code with detailed docstrings
* **Issues**: Report bugs or request features on the project repository

Contributing
------------------------

* **Code Style**: Follow PEP 8 and use Google Style docstrings
* **Documentation**: Keep docstrings up to date with code changes
* **Testing**: Add tests for new functionality
* **Examples**: Provide examples for new features
