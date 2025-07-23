# Changelog

This file documents the changes to the features for supported feature versions.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [0.1.0] - 2025-07-23 - [#22](https://github.com/int-brain-lab/ibleatools/pull/22)

### Added
- `ephysatlas.reaveal.AtlasReveal` class to create figures of the feature extraction and prediction on the website.

### Modified
 - When aggregating `spike_count` features across multiple time snippets for a probe insertion, NaN values are replaced with zeros before calculating the mean value.
 - For `channel_labels` , mode is used to do the aggregation across snippets.
 - For rest of the features, we do the aggregation using nanmedian. 
 