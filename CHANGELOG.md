# Changelog

This file documents the changes to the features for supported feature versions.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [UNRELEASED]

### Changed
**Cells Features**
- `compute_log_acg`: simplified implementation using `np.geomspace` directly in time-space; removed `log_start` parameter; output length is now exactly `n_log_bins` (previously variable after trimming)

## [0.6.0] - 2026-05-22

### Added
**Cells Features***
- Unit tests for `compute_burstiness_and_memory` in `tests/test_cells.py`

**Schemas**
- `ModelClusters` pandera schema in `ephysatlas.cells` for cluster-level features (good_clusters.pqt / all_clusters.pqt)
- `ModelProbeDetails` pandera schema in `ephysatlas.features` for probe insertion metadata (df_probe_details.pqt)

**Data access**
- `download_probe_details()`, `download_cell_features()`, and `download_project_data()` in `ephysatlas.data` for fetching project data from S3; probe details and cell aggregates are separate calls to avoid downloading ~1 GB unnecessarily
- `read_probe_details()` and `read_cell_features()` in `ephysatlas.data` for loading project data from disk with optional pandera validation

**Tests**
- `TestProjectDataIO` test class with synthetic fixtures generated from pandera schemas

### Fixed
- Fixed `Series[T]` annotations in all pandera `DataFrameModel` subclasses (`ChannelDataFrameSchema`, `ModelLfFeatures`, `ModelCsdFeatures`, `ModelApFeatures`, `ModelSpikeFeatures`, `ModelChannelLayout`, `ModelHistologyResolved`) to use plain Python types, required by pandera 0.25.0

## [0.5.0]

### Added
- Added `replace_nan` utility function in `ephysatlas.data` module for replacing NaN values with median in feature dataframes
- Added `ibl-neuropixel==1.9.1` dependency (pinned to avoid compatibility issues with ibldsp.cazdow filter)
- Added comprehensive metadata (descriptions, raw units, transformed units) to all schema field definitions in feature models

### Modified
- Modified `denoise_raw_features_data` to apply `replace_nan` after outlier treatment to handle remaining NaN values
- Modified `compute_features_from_pid` merge logic to handle both `rawInd` and `channel` column names in channels dictionary (temporary fix)

### Fixed
- Fixed assert message path in `download_tables` function (changed from `aggregates/atlas/{project}/{label}` to `aggregates/atlas/features/{project}/{label}`)
- Fixed edge case in `get_psd_decay_features` to handle channels with zero-sum PSD by returning NaN values
- Fixed edge case in `denoise_shank` to handle cases with no valid data points
- Fixed `EphysTransformer.transform` to properly check for transform metadata before accessing it

## [0.4.0]

### Added
- Added `outlier_treatment` utility function in `ephysatlas.data` module for handling outlier channels in feature dataframes
- Added `denoise` parameter to `infer_regions` function to optionally apply denoising during inference
- Added `project` parameter to `atlas_pids` function to allow querying different IBL projects

### Modified
- Modified `EphysTransformer.transform` to preserve columns that are not in the transformation dictionary
- Modified `EphysDenoiser.fit_transform` to preserve original feature dtypes after denoising
- Modified `compute_features_from_pid` to return dataframe merged with channel information
- Modified `get_aggregated_features_per_pid` and `denoise_raw_features_data` to use centralized `outlier_treatment` function
- Modified `plot_results` to use updated model structure (accessing `FEATURES` and `CLASSES` directly from model dict)
- Modified `figure_features_channel_space` to handle cases where brain regions are not available and use sklearn config context for NaN handling
- Modified `get_color_feat` to use `np.nanmin` and `np.nanmax` instead of `np.min` and `np.max` for better NaN handling

### Fixed
- Fixed model loading structure in `infer_regions` to correctly unpack classifier and model_info from `load_model`

## [0.3.0]

### Added
- Added SDSC utils for generating task files.
- Added luigi dependency for workflow management
- Added bad alpha filtering in aggregation pipeline to improve data quality

### Fixed
-   `features.voltage_features_set` returns features by categories, sorted as the pydantic model definitions
## [0.2.2] - 2025-09-25

### Added
 - Added utility functions for listing the latest labels for features data on AWS.
 - Added new LF Features related to the slope and intercept of the PSD decay.
 - Migrated from pip to uv for the Github CI.
 - Added specparam as a dependency for the spectral parameterization analysis.

### Modified
 - Modified the `download_tables` function in `ephysatlas.data` module. The specific project and aggregation level can be provided now to the function. Also now the local path is created if it does not exists.


## [0.2.1] - 2025-07-27

### Added
- Added Sphinx documentation system
- How-to guides for common tasks
- Installation and configuration documentation
- Using Google docstring format everywhere now
- Sphinx build configuration and Makefile for documentation generation
- Added True label score in the reveal figure.


## [0.2.0] - 2025-07-23 - [#23](https://github.com/int-brain-lab/ibleatools/pull/23)

### Added
- the transform and denoise phases are distinct:
  - `ephysatlas.features.EphysDenoiser`: scikit-learn transformer interface for total variation denoising of features
  - `ephysatlas.features.EphysTransformer`: scikit-learn transformer interface for feature transformation

### Fixed
- `ephysatlas.features.voltage_features_set`: the order of panderas schemes is not stable, make sure it is sorted.


## [0.1.0] - 2025-07-23 - [#22](https://github.com/int-brain-lab/ibleatools/pull/22)

### Added
- `ephysatlas.reveal.AtlasReveal` class to create figures of the feature extraction and prediction on the website.

### Modified
 - When aggregating `spike_count` features across multiple time snippets for a probe insertion, NaN values are replaced with zeros before calculating the mean value.
 - For `channel_labels` , mode is used to do the aggregation across snippets.
 - For rest of the features, we do the aggregation using nanmedian. 
 