# Features Changelog

The features are released when necessary with the following tag: yyyy_Www
2024_W50 means it has been released on the week 50 of 2024.

This file documents the changes to the features for supported feature versions.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2025_W30]
### Modified
 - When aggregating `spike_count` features across multiple time snippets for a probe insertion, NaN values are replaced with zeros before calculating the mean value.
 - For `channel_labels` , mode is used to do the aggregation across snippets.
 - For rest of the features, we do the aggregation using nanmedian. 
 
## [2025_W28]

### Modified
- denoised dataframe applies denoising in log space for `rms_ap` (dB), `rms_lf` (dB) and `spike_count` (log2), and returns those features transformed accordingly.
- channels `x_target`, `y_target` and `z_target` have a pitch correction of 5 degrees applied to account for the difference between Allen reference and in-vivo head position.

### Added
- features list:
  - contain the `channel_labels` field with the ibl-neuropixel bad channel flag. 


## [2024_W50]

### Added
- features list:
    - alpha_mean
    - alpha_std
    - cor_ratio
    - depolarisation_slope
    - peak_time_secs
    - peak_val
    - polarity
    - psd_alpha
    - psd_alpha_csd
    - psd_beta
    - psd_beta_csd
    - psd_delta
    - psd_delta_csd
    - psd_gamma
    - psd_gamma_csd
    - psd_lfp
    - psd_lfp_csd
    - psd_theta
    - psd_theta_csd
    - recovery_slope
    - recovery_time_secs
    - repolarisation_slope
    - rms_ap
    - rms_lf
    - rms_lf_csd
    - spike_count
    - tip_time_secs
    - tip_val
    - trough_time_secs
    - trough_val
- channels list:
    - acronym
    - atlas_id
    - atlas_id_target
    - axial_um
    - lateral_um
    - x
    - x_target
    - y
    - y_target
    - z
    - z_target
