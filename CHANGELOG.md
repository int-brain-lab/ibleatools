# Changelog

This file documents the changes to the features for supported feature versions.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `ephysatlas.features.ModelSpikeSharedFeatures` gains `spatial_spread_um_std` and `slowness_s_per_m_std`, the across-spike standard deviations of the two multi-channel waveform features #125 added, aggregated per channel in `features.spikes` next to their existing means. Both are nullable and use the sample standard deviation (ddof=1) over the spikes with a usable value, so a channel with fewer than two of those reports NaN rather than a misleading `0.0`. Being on the shared base class they reach both schemas: `ModelSpikeFeatures` goes from 16 to 18 columns and `ModelSpikeShapeFeatures` from 14 to 16, and both `remap_waveform_shape_features` and its `_volume` counterpart pass them through unchanged, treating them as all-NaN when absent from pre-#127 data (#127)

## [0.9.0] - 2026-09-20

### Added
- `ephysatlas.features.ModelSpikeShapeFeatures`: a sparser, neuroscientist-facing spike-shape schema that reparametrises `ModelSpikeFeatures`'s 14 raw waveform columns into 12. Four are new and directly interpretable -- `spike_width_secs` (trough-to-peak duration), `predepolarisation_width_secs` (tip-to-peak duration), `spike_amplitude` (trough-to-peak amplitude) and `peak_to_trough_ratio_log` (amplitude-independent shape asymmetry). Eight pass through unchanged (`tip_val`, `alpha_mean`, `alpha_std`, `polarity`, `spike_count` and the three slopes). Six are dropped as redundant: `peak_val`, `trough_val`, `peak_time_secs`, `trough_time_secs` and `tip_time_secs` are reparametrised into the four new columns without losing relative information, and `recovery_time_secs` is an exact constant offset of `trough_time_secs`. PCA over the raw set needs only ~7-8 components for 95% of its variance, which motivated the reduction (#118)
- `ephysatlas.features.remap_waveform_shape_features(df)`: applies the remapping to a channel-level dataframe, returning `ModelSpikeShapeFeatures` columns on the same index (#118)
- `ephysatlas.features.remap_waveform_shape_features_volume(vol, feature_names)`: the same transform for a `download_encoding_volume` array, returning the remapped volume and its feature names. Both entry points share one container-agnostic implementation, so the dataframe and volume paths cannot drift (#118)

### Changed
- **Breaking.** Waveform amplitudes are now in Volts. `features.dart_subtraction_numpy` un-applies the per-channel RMS normalisation DARTsort applies internally (its detection threshold is expressed in units of channel RMS), which was previously left in place: `df_spikes["ptp"]` and the `raw` / `denoised` waveform snippets are rescaled back to Volts, and with them every amplitude derived from them downstream -- `peak_val`, `trough_val`, `tip_val` and the three slopes (now V/s). Every caller gets different values; the old ones were dimensionless channel-RMS units. `alpha_mean` / `alpha_std` are the exception: DARTsort fits the localisation amplitude jointly across a multichannel snippet whose channels each carry a different RMS, so no single factor recovers Volts and they stay in DARTsort's native, non-physical units. The per-channel RMS used for the rescaling is returned as a new `channel_rms` key on the waveforms dict, which therefore has 5 keys instead of 4 (#124)
- **Breaking.** `spike_count` is a spike rate in spikes/s, not a raw per-snippet count: `features.spikes` divides by the snippet duration, so the value no longer depends on the snippet length (e.g. the 5 s AP default). `feature_computation.compute_features_from_destriped` no longer casts it to `Int64`, and the schema metadata becomes `raw_unit` `count/s` / `transformed_unit` `log2 (count/s)`; the log2 transform itself is unchanged (#124)
- `DEFAULT_FAC["waveforms"]` is 3, not the 5 that 0.8.0 shipped -- the bracketing analysis in #112 supports 3 (#124)
- `ephysatlas.features.ModelSpikeSharedFeatures`: the 8 fields `ModelSpikeFeatures` and `ModelSpikeShapeFeatures` declared identically (`alpha_mean`, `alpha_std`, `polarity`, `spike_count`, `tip_val` and the three slopes) now live on one shared base class that both inherit from, so a unit or description fix lands in one place. The column *sets* of both schemas are unchanged; `ModelSpikeShapeFeatures`'s column *order* changes -- the inherited fields come first -- and `remap_waveform_shape_features` follows it (#124)
- `features.lf`, `features.csd`, `features.ap` and `features.spikes` return the pandera-validated dataframe instead of the pre-validation one, so the schemas' `coerce=True` dtypes now actually reach the caller (#118)
- `sdsc_utils.parse_logs_to_dataframe` matches a trajectory failure on `"trajectory found for pid"`, the message `add_target_coordinates` now raises, instead of the removed `Micro-manipulator` generator expression (#118)

### Fixed
- `feature_computation.add_target_coordinates` no longer fails on insertions that have neither a micro-manipulator nor a planned trajectory. The Alyx query is widened from `provenance__lte,30` to `provenance__lte,50` and trajectories are chosen by the new `feature_computation.PROVENANCE_PREFERENCE`: micro-manipulator, else planned, else histology track. Insertions that resolved before resolve to the same trajectory as before, so `x_target` / `y_target` / `z_target` are unchanged for them; insertions with none of the three now raise `ValueError` instead of `IndexError`. Example of an insertion this fixes: `1a924329-65aa-465d-b201-c2dd898aebd0`
- A histology track trajectory is read in the Allen frame Alyx stores it in (`coordinate_system` `IBL-Allen`), skipping the -5 degree pitch correction and the needles -> Allen conversion that only apply to planned and micro-manipulator trajectories (`Needles-Allen`). Putting one through the needles path displaces the channels by ~320-460 um
- `feature_computation.add_target_coordinates` no longer writes the tilt-corrected `theta` / `phi` back into the caller's `traj_dict`, which made a second call on the same dictionary apply the -5 degree correction twice
- The `x` / `y` / `z` coordinate columns are documented and tagged as **metres** from Bregma (IBL coordinate space), not micrometres: `ChannelDataFrameSchema`, `ModelHistologyResolved` and `ModelHistologyPlanned`'s `x_target` / `y_target` / `z_target` all move their `raw_unit` metadata from `um` to `m`. The stored values were always metres, so no data changes -- only the descriptions and the machine-readable units were wrong. Note the two length units legitimately coexist in one table: `axial_um` / `lateral_um` really are micrometres, being probe geometry rather than atlas coordinates (#119)

## [0.8.0] - 2026-09-15

### Added
- Public inference API: `ephysatlas.load_pretrained(model_id, revision=..., cache_dir=...)` is now the single entry point for every published model, taking either a Hugging Face repo id (`owner/name`) or a local directory. Dispatch is driven by the model's `ephysatlas_model.json` manifest, artifact checksums are verified on every load (local directories included), and every wrapper exposes the same `.predict()` / `.selftest()` / `.index` contract. Wrappers are imported lazily, so `import ephysatlas` pulls in neither torch nor xgboost (#108)
- `ephysatlas.model_registry`: `resolve_model`, `read_manifest`, `meta_from_manifest`, `class_acronyms`, `verify_checksums`, `feature_order_sha256` / `validate_feature_order`, and `HFModelSource`. Models are fetched from public Hugging Face repos and need no ONE account or AWS credentials. A mismatch between the caller's feature order and the published `feature_order_sha256` raises `ValueError` rather than silently producing wrong predictions (#108)
- `RegionClassifier` in `ephysatlas.regionclassifier`: channel -> brain region via the XGBoost fold ensemble, with `.from_pretrained()`, `.predict(df, denoise=False, estimator="ensemble"|"global")` and `.selftest()`. `.predict()` returns namespaced columns `predicted_acronym`, `predicted_atlas_id`, `prediction_probability`, `fold_agreement` and one `p_<acronym>` per class, so the result joins onto a channel table that already carries histology `acronym` / `atlas_id` without colliding (#108)
- `SpatialEncoder` in `ephysatlas.models.encoder_inpainting`: the inverse direction -- channel position (`x`, `y`, `z`) plus anatomical context -> ephys features, by neighbour inpainting. `.predict()` returns one `pred_<feature>` column per output feature; `.preprocessing_stats()`, `.context_dir` and `.confidence_model()` expose the published standardisation buffers, PCA context volumes and probe-confidence checkpoint. `build_neighbor_bank()` covers the publication side (#109)
- `UnitEncoder` in `ephysatlas.models.unit_encoder`, with the supporting `ephysatlas.unit_level_encoder` package: spike-sorted units -> a 32-d phenotype latent from waveform + autocorrelogram (`MultimodalAutoencoder`), plus a `PointTransformerGMM` over those latents whose components read as putative cell types. `.encode()`, `.reconstruct()`, `.components()`, `.assign()`, `.latents()`, `.atlas_arrays()` (#110)
- `ProbeTransformerClassifier` and `ProbeTransformer` in `ephysatlas.models.probe_transformer`: whole-probe channel -> region prediction with a RoPE transformer attending across all channels of a probe, using `axial_um` as the positional signal. A release ships one model per random seed, so `.predict()` reports `seed_agreement` -- not the `fold_agreement` of the XGBoost classifier. Channels with any non-finite feature are dropped with a warning, so the result may be a subset of the input (#117)
- `examples/inference_region_classifier_public.py`: reference usage of the public API, deliberately using no ONE account, no AWS credentials and no raw data (#108)
- `split_manifest`, `preprocessing_stats` and `return_preprocessing_stats` keyword arguments on `build_channels_plus_emptyvoxels_with_neighbors` (`ephysatlas.spatial_encoder.utils`), so evaluation can reuse a released model's own train/validation/test split and its train-time standardisation instead of recomputing either. All three default to the previous behaviour (#109)
- `huggingface_hub>=0.34` dependency (#108)
- `tests/test_repo_segregation.py`: enforces that this repo stays the public inference surface -- importing the public API must pull in nothing under `examples/` (or a re-introduced `training/`), and must not import torch or xgboost at module scope, which segfault together on macOS arm64 (#108)

### Changed
- **Breaking.** `regionclassifier.download_model(local_path, model_name, revision=None)` no longer takes `one` or `overwrite`, and fetches from the public Hugging Face Hub instead of IBL's AWS S3 via ONE/Alyx. `model_name` is now a Hub repo id such as `int-brain-lab/ea-decoder-channel-xgboost`, not a vintage directory name like `2024_W50_Cosmos_lid-basket-sense` (#108)
- **Breaking.** `regionclassifier.load_model` reads the publication contract from `ephysatlas_model.json` instead of `meta.yaml`, and raises `FileNotFoundError` on a directory with no manifest and `ValueError` on an unregistered `model_class`. The `(classifier, model_info)` return shape is unchanged and `model_info` keeps its UPPER_CASE keys (`FEATURES`, `CLASSES`, ...), now projected from the manifest by `meta_from_manifest` (#108)
- **Breaking.** `regionclassifier.infer_regions` ignores `n_folds`: the fold count and layout come from the manifest (`folds/FOLD0k/model.ubj`, previously hardcoded `FOLD0k/`), and a value disagreeing with the manifest logs a warning. The returned `predicted_region` is now the argmax class index in manifest class order rather than the output of `classifier.predict()`, and denoising is applied once before the fold loop instead of being re-applied inside it (#108)
- TV denoising accepts a per-feature-group `fac`: `EphysDenoiser`, `denoise_dataframe` and `denoise_raw_features_data` take either a scalar (unchanged behaviour) or a dict keyed by `raw_ap` / `raw_lf` / `raw_lf_csd` / `waveforms`; groups absent from the dict keep a weight of 1. **The default changed** from the scalar `fac=1` to `DEFAULT_FAC = {raw_ap: 0.1, raw_lf: 0.1, raw_lf_csd: 0.1, waveforms: 5}`, selected by a bracketing sweep against Cosmos_id classification accuracy and confirmed with 5-fold CV. Callers that never passed `fac` explicitly will get different denoised values (#112)
- The `[full]` extra requires `dartsort>=0.5.24` (was `>=0.5.23`): 0.5.24 changed `FeaturizationConfig.do_enforce_decrease` from a bool to `Literal["yes", "no", "loc_only"]`, and `features.dart_subtraction_numpy` now passes `"yes"`
- CI runs tests on pull requests against any base branch, so a PR stacked on another feature branch is no longer skipped (#116)

### Removed
- **Breaking.** `regionclassifier.save_model`, and with it the `meta.yaml` publication convention. Region-classifier *training* is no longer part of this package -- it lives in `paper-ephys-atlas` (the `ephys_atlas` package). `examples/training_region_predictor_gradient_boosting.py` is deleted and `tests/test_repo_segregation.py` enforces the boundary. `tests/test_region_classifier.py` loses its `save_model` round-trip test; its `TestViterbi` coverage is unchanged (#108)

### Known issues
- `examples/inference_spatial_encoder.py` has not been migrated to the public inference API: it still calls `download_model(..., one=one)`, whose `one` argument was removed in this release, and fails as written. The supported path is `load_pretrained(...).predict(df)` with channel positions. The file logs a warning to this effect on run (#109). This file will be updated in a future PR.

## [0.7.0] - 2026-09-08

### Added
- Object-oriented feature-computation layer in `ephysatlas.feature_calculators`: `BaseFeatureCalculator`, `SpikeGlxLikeFeatureCalculator`, and the concrete `IBLPIDFeatureCalculator` (ONE/SpikeSortingLoader) and `SpikeGLXFileFeatureCalculator` (local AP/LF files). `compute_snippet()` is the shared template that both public entry points now use.
- Typed per-feature parameters `FeatureParams` / `LfParams` / `CsdParams` (in `ephysatlas.feature_calculators`), settable on `compute_features_from_pid` / `compute_features_from_file` and forwarded to the engine. Accepts either the dataclasses or a nested dict (e.g. `{"csd": {"scale": False}}`), which is validated and normalized to `FeatureParams`. This makes per-feature options such as the CSD `scale` flag configurable end to end.
- `destripe_ap_lf()` and `compute_features_from_destriped()` in `ephysatlas.feature_computation`: `compute_features_from_raw` now destripes via the shared `destripe_ap_lf` primitive and delegates feature computation to `compute_features_from_destriped`, which can also be called directly on already-destriped data.
- OOP usage examples: `examples/feature_extraction_example.py` (public `compute_features_from_pid` path) and `examples/feature_extraction_oop.py` (`IBLPIDFeatureCalculator` / `SpikeGLXFileFeatureCalculator` directly).
- `rms_lf_no_car` LF feature: RMS of the LF band destriped without common-average referencing (`k_filter=None`), opt-in via `feature_params.lf.compute_rms_no_car` (default off).
- `scale` parameter (default `True`) on `ephysatlas.features.csd`, surfaced through `CsdParams.scale`, controlling whether the CSD is scaled.
- Published the Sphinx documentation to GitHub Pages via a `.github/workflows/documentation.yaml` Actions workflow (builds on pushes to `main` and pull requests; deploys from `main`).
- `ModelProbeDetails` gains `probe_model` and `referencing_scheme` columns, populated from SpikeGLX meta-data via the new `ibl-neuropixel` `spikeglx.get_probe_model` / `spikeglx.get_referencing_scheme` (#82)
- `enrich_channel_metadata` now broadcasts `probe_model` and `referencing_scheme` onto every row of the channel metadata for both SpikeGLX-backed sources (`SpikeGLXFileFeatureCalculator` and `IBLPIDFeatureCalculator`), read from the first reader (AP, then LF) carrying SpikeGLX meta-data (#82)
- The DARTsort subtraction parameters `detection_threshold`, `spatial_dedup_radius_um`, `positive_temporal_dedup_radius_samples` and `residnorm_decrease_threshold` are settable via `DartParameters` and `FeatureParams.waveforms`; defaults are unchanged (#106)
- `ephysatlas.cells.spike_triggered_population_coupling` / `spike_triggered_population_coupling_df`: FFT-based spike-triggered population coupling matching the Methods of Bimbard, Harris & Carandini 2025 (bioRxiv 2025.12.20.695676). Replaces the previous implementation, kept as `spike_triggered_population_coupling_windowed` / `get_neighbours_members_windowed`, a distinct windowed/overlapping-correlation estimator (#95)

### Changed
- Channel metadata is now merged onto the feature table on the physical recording site `(axial_um, lateral_um, shank)` (rounded to the nearest micrometre) instead of on `channel` / `rawInd`, whose numbering is unreliable across data sources. `rawInd` is carried as descriptive metadata only. Channel builders now always provide `shank`.
- `compute_features_from_pid` and `compute_features_from_file` are thin wrappers that delegate to the OOP calculators (`compute_features_from_raw` engine). The on-disk layout for the file path is now named after the AP/LF file stem (previously an md5 hash / `probe_unknown_pid_*`); the returned DataFrame, `channels.pqt`, and per-snippet `.attrs` are unchanged.
- De-duplicated the reader-contract logic (`load_raw_snippet`, geometry-default filling, `_resolve_channel_labels`) into the shared `SpikeGlxLikeFeatureCalculator`; `get_destriped_snippet` now delegates to `destripe_ap_lf` so the debug/inspection path cannot diverge from the production destriping.
- The `[full]` extra installs DARTsort from PyPI (`dartsort>=0.5.23`) instead of the pinned `iblsorter` git fork, no longer pins DREDge directly (DARTsort installs `dredge-ephys` itself), and requires `spikeinterface>=0.104.0`. `ephysatlas.features.dart_subtraction_numpy` is adapted to the DARTsort 0.5 API. Waveform geometry is unchanged (121 samples, trough at sample 42), but detections increase ~9.5% and amplitudes are ~10% smaller (#106)
- Minimum supported Python is now 3.11, required by DARTsort 0.5 (#106)

### Deprecated
- `compute_features`, `online_feature_computation`, and `load_data_from_pid` in `ephysatlas.feature_computation` are deprecated in favor of `compute_features_from_pid` / `compute_features_from_file` (and the OOP calculators) and will be removed in a future version.

### Fixed
- `spike_triggered_population_coupling_windowed`: correct for the 2x redundancy introduced by its overlapping (50%-hop) accumulation windows, previously left uncorrected in the raw `stpc` magnitude

## [0.6.0]

### Changed
- Cadzow denoising updated to `cadzow_denoiser` (replaces deprecated `cadzow_np1`): uses batched SVD, geometry-agnostic; `rank=5`, `fmax=125`, `nswx=64`, `ovx=32`, `gap_threshold=2.0`, `ppca_k=2.0`
- Bumped `ibl-neuropixel` requirement to `>=1.11.0`

### Added
**Cells Features***
- Unit tests for `compute_burstiness_and_memory` in `tests/test_cells.py`
- `compute_log_acg`: simplified implementation using `np.geomspace` directly in time-space; removed `log_start` parameter; output length is now exactly `n_log_bins` (previously variable after trimming)

**Schemas**
- `ModelClusters` pandera schema in `ephysatlas.cells` for cluster-level features (good_clusters.pqt / all_clusters.pqt)
- `ModelProbeDetails` pandera schema in `ephysatlas.features` for probe insertion metadata (df_probe_details.pqt)

**Data access**
- `download_probe_details()`, `download_cell_features()`, and `download_project_data()` in `ephysatlas.data` for fetching project data from S3; probe details and cell aggregates are separate calls to avoid downloading ~1 GB unnecessarily
- `download_cells_features` and `download_project_data` gain a `large_files=False` parameter; `waveforms.voltage.npy` and `waveforms.table.pqt` (~8 GB) are now opt-in and excluded by default
- `read_cells_features` returns `waveforms` / `df_waveforms` keys only when the files are present on disk (downloaded with `large_files=True`), avoiding a hard crash for the common case
- `read_probe_details()` and `read_cell_features()` in `ephysatlas.data` for loading project data from disk with optional pandera validation

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
 