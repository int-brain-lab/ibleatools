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
* **Data Visualization**: Publication-ready plotting tools for probe data and brain regions
* **Brain Region Classification**: Machine learning models for automatic brain region identification
* **Data Management**: Utilities for organizing and managing large-scale electrophysiological datasets

Key Features
------------

* **Multi-backend Support**: Spike detection using Dartsort or SpikeInterface
* **Automated Processing**: Streamlined pipelines for feature computation and analysis
* **Quality Control**: Bad channel detection and data validation
* **Flexible Output**: Support for various data formats and storage options

Quick Start
------------

Get started with basic feature extraction:

.. code-block:: python

   from ephysatlas.feature_computation import compute_features_from_pid
   from one.api import ONE

   # Initialize ONE client
   one = ONE()

   # Compute features for a probe insertion
   pid = "0228bcfd-632e-49bd-acd4-c334cf9213e9"  # Example probe ID
   features = compute_features_from_pid(
       pid=pid,
       one=one,
       t_start=300,  # Start at 300 seconds
       duration=5,   # Compute 5 seconds
       output_dir="./features"  # Save features to directory
   )

Installation
-------------

Install ibleatools in editable mode:

.. code-block:: bash

   git clone https://github.com/int-brain-lab/ibleatools.git
   cd ibleatools
   pip install -e .

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