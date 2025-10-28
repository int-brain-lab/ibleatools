# Features Changelog

The features are released when necessary with the following tag: yyyy_Www
2024_W50 means it has been released on the week 50 of 2024.

This file documents the changes to the features for supported feature versions.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2025_W43]
  - Added bad alpha filtering in aggregation pipeline to improve data quality

## [2025_W39]

### Added
 - Added spectral parameterization features: New PSD decay analysis using the specparam library
     - Aperiodic component features:
        - aperiodic_offset: Y-intercept of the 1/f component
        - aperiodic_exponent: Slope of the aperiodic component
        - decay_fit_error: RMS error of spectral model fit
        - decay_fit_r_squared: Goodness of fit metric
        - decay_n_peaks: Number of detected periodic peaks
     - Residual power features: Residual power in various frequency bands after removing aperiodic component in the log space.
        - psd_residual_delta, psd_residual_theta, psd_residual_alpha, psd_residual_beta, psd_residual_gamma, psd_residual_lfp


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
