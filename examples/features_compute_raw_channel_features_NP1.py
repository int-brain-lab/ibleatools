# %%
import spikeglx
import ephysatlas.feature_computation


file_ap = "/mnt/s1/spikesorting/raw_data/mrsicflogellab/Subjects/SWC_038/2020-07-30/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin"
file_lf = "/mnt/s1/spikesorting/raw_data/mrsicflogellab/Subjects/SWC_038/2020-07-30/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.lf.cbin"

t0 = 485

sr_ap = spikeglx.Reader(file_ap)
sr_lf = spikeglx.Reader(file_lf)
duration = 3

df_features = ephysatlas.feature_computation.online_feature_computation(
    sr_lf, sr_ap, t0, duration, channel_labels=None
)
