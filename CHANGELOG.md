# Changelog

This file documents the changes to the features for supported feature versions.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


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
 