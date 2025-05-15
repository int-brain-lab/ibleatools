import logging
from pathlib import Path
import yaml
import hashlib

import numpy as np
import pandas as pd
from ibldsp.waveforms import peak_to_trough_ratio
import neuropixel
from one.remote import aws
from iblutil.numerical import ismember

from one.api import ONE
from iblatlas.atlas import Insertion, NeedlesAtlas, AllenAtlas, BrainRegions, tilt_spherical
from ibllib.pipes.histology import interpolate_along_track


_logger = logging.getLogger("ibllib")

SPIKES_ATTRIBUTES = ["clusters", "times", "depths", "amps"]
CLUSTERS_ATTRIBUTES = ["channels", "depths", "metrics"]

EXTRACT_RADIUS_UM = 200  # for localisation , the default extraction radius in um

BENCHMARK_PIDS = [
    "1a276285-8b0e-4cc9-9f0a-a3a002978724",
    "1e104bf4-7a24-4624-a5b2-c2c8289c0de7",
    "5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e",
    "5f7766ce-8e2e-410c-9195-6bf089fea4fd",
    "6638cfb3-3831-4fc2-9327-194b76cf22e1",
    "749cb2b7-e57e-4453-a794-f6230e4d0226",
    "d7ec0892-0a6c-4f4f-9d8f-72083692af5c",
    "da8dfec1-d265-44e8-84ce-6ae9c109b8bd",
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0",
    "dc7e9403-19f7-409f-9240-05ee57cb7aea",
    "e8f9fba4-d151-4b00-bee7-447f0f3e752c",
    "eebcaf65-7fa4-4118-869d-a084e84530e2",
    "fe380793-8035-414e-b000-09bfe5ece92a",
]

# this is from the content of the 2024-03 freeze csv file on the repository
# https://github.com/int-brain-lab/paper-reproducible-ephys/blob/master/data_releases/freeze_2024_03.csv
REPRO_EPHYS_PIDS = [
    "02cc03e4-8015-4050-bb42-6c832091febb",
    "0851db85-2889-4070-ac18-a40e8ebd96ba",
    "0aafb6f1-6c10-4886-8f03-543988e02d9e",
    "0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a",
    "11a5a93e-58a9-4ed0-995e-52279ec16b98",
    "143dd7cf-6a47-47a1-906d-927ad7fe9117",
    "16799c7a-e395-435d-a4c4-a678007e1550",
    "17d9710a-f292-4226-b033-687d54b6545a",
    "19baa84c-22a5-4589-9cbd-c23f111c054c",
    "1a60a6e1-da99-4d4e-a734-39b1d4544fad",
    "1e176f17-d00f-49bb-87ff-26d237b525f1",
    "1f3d3fcb-f188-47a2-87e5-ac1db6cf393a",
    "1f5d62cb-814f-4ab2-b6af-7557ea04d56a",
    "22212d26-a167-45fb-9963-35ecd003e8a2",
    "27bac116-ea57-4512-ad35-714a62d259cd",
    "31f3e083-a324-4b88-b0a4-7788ec37b191",
    "341ef9bb-25f9-4eeb-8f1d-bdd054b22ba8",
    "36362f75-96d8-4ed4-a728-5e72284d0995",
    "3d3d5a5e-df26-43ee-80b6-2d72d85668a5",
    "3fded122-619c-4e65-aadd-d5420978d167",
    "41a3b948-13f4-4be7-90b9-150705d39005",
    "44fc10b1-ec82-4f88-afbf-10a6f8a1b5d8",
    "478de1ce-d7e7-4221-9365-2abdc6e88fb6",
    "4836a465-c691-4852-a0b1-dcd2b1ce38a1",
    "485b50c8-71e1-4654-9a07-64395c15f5ed",
    "49c2ea3d-2b50-4e8a-b124-9e190960784e",
    "4b93a168-0f3b-4124-88fa-a57046ca70e1",
    "523f8301-4f56-4faf-ab33-a9ff11331118",
    "5246af08-0730-40f7-83de-29b5d62b9b6d",
    "57656bee-e32e-4848-b924-0f6f18cfdfb1",
    "63517fd4-ece1-49eb-9259-371dc30b1dd6",
    "63a32e5c-f63a-450d-85cb-140947b67eaf",
    "69f42a9c-095d-4a25-bca8-61a9869871d3",
    "6b6af675-e1ef-43a6-b408-95cfc71fe2cc",
    "6d9b6393-6729-4a15-ad08-c6838842a074",
    "6e1379e8-3af0-4fc5-8ba8-37d3bb02226b",
    "6fc4d73c-2071-43ec-a756-c6c6d8322c8b",
    "70da415f-444d-4148-ade7-a1f58a16fcf8",
    "72f89097-4836-4b67-a47a-edb3285a6e83",
    "730770d6-617a-4ada-95db-a48521befda5",
    "7a620688-66cb-44d3-b79b-ccac1c8ba23e",
    "7cbecb3f-6a8a-48e5-a3be-8f7a762b5a04",
    "7d999a68-0215-4e45-8e6c-879c6ca2b771",
    "7f3dddf8-637f-47bb-a7b7-e303277b2107",
    "80624507-4be6-4689-92df-0e2c26c3faf3",
    "80f6ffdd-f692-450f-ab19-cd6d45bfd73e",
    "8185f1e9-cfe0-4fd6-8d7e-446a8051c588",
    "82a42cdf-3140-427b-8ad0-0d504716c871",
    "8413c5c6-b42b-4ec6-b751-881a54413628",
    "84bb830f-b9ff-4e6b-9296-f458fb41d160",
    "84fd7fa3-6c2d-4233-b265-46a427d3d68d",
    "8abf098f-d4f6-4957-9c0a-f53685db74cc",
    "8b7c808f-763b-44c8-b273-63c6afbc6aae",
    "8c732bf2-639d-496c-bf82-464bc9c2d54b",
    "8ca1a850-26ef-42be-8b28-c2e2d12f06d6",
    "8d59da25-3a9c-44be-8b1a-e27cdd39ca34",
    "9117969a-3f0d-478b-ad75-98263e3bfacf",
    "92033a0c-5a14-471b-b131-d43c72ca5d7a",
    "92822789-608f-44a6-ad64-fe549402b2df",
    "94e948c1-f7be-4868-893a-f7cd2df3313e",
    "9657af01-50bd-4120-8303-416ad9e24a51",
    "9e44ddb5-7c7c-48f1-954a-6cec2ad26088",
    "a12c8ae8-d5ad-4d15-b805-436ad23e5ad1",
    "a3d13b05-bf4d-427a-a2d5-2fe050d603ec",
    "a8a59fc3-a658-4db4-b5e8-09f1e4df03fd",
    "ad714133-1e03-4d3a-8427-33fc483daf1a",
    "ae252f7b-0224-4925-8174-7b25c2385bb7",
    "b25799a5-09e8-4656-9c1b-44bc9cbb5279",
    "b2746c16-7152-45a3-a7f0-477985638638",
    "b749446c-18e3-4987-820a-50649ab0f826",
    "b83407f8-8220-46f9-9b90-a4c9f150c572",
    "bbe6ebc1-d32f-42dd-a89c-211226737deb",
    "bc1602ba-dd6c-4ae4-bcb2-4925e7c8632a",
    "bf96f6d6-4726-4cfa-804a-bca8f9262721",
    "c07d13ed-e387-4457-8e33-1d16aed3fd92",
    "c17772a9-21b5-49df-ab31-3017addea12e",
    "c4f6665f-8be5-476b-a6e8-d81eeae9279d",
    "c6e294f7-5421-4697-8618-8ccc9b0269f6",
    "ca5764ea-a57e-49de-8156-84da18ad439f",
    "ce397420-3cd2-4a55-8fd1-5e28321981f4",
    "d004f105-9886-4b83-a59a-f9173131a383",
    "d7361c6f-6751-4b5f-91c2-fdd61f988aa4",
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0",
    "dc50c3de-5d84-4408-9725-22ae55b93522",
    "e31b4e39-e350-47a9-aca4-72496d99ff2a",
    "e42e948c-3154-45cb-bf52-408b7cda0f2f",
    "e49f221d-399d-4297-bb7d-2d23cc0e4acc",
    "e4ce2e94-6fb9-4afe-acbf-6f5a3498602e",
    "ee3345e6-540d-4cea-9e4a-7f1b2fb9a4e4",
    "eeb27b45-5b85-4e5c-b6ff-f639ca5687de",
    "ef03e569-2b50-4534-89f4-fb1e56e81461",
    "ef3d059a-59d5-4870-b355-563a8d7cfd2d",
    "f03b61b4-6b13-479d-940f-d1608eb275cc",
    "f26a6ab1-7e37-4f8d-bb50-295c056e1062",
    "f2a098e7-a67e-4125-92d8-36fc6b606c45",
    "f2ee886d-5b9c-4d06-a9be-ee7ae8381114",
    "f4bd76a6-66c9-41f3-9311-6962315f8fc8",
    "f68d9f26-ac40-4c67-9cbf-9ad1851292f7",
    "f86e9571-63ff-4116-9c40-aa44d57d2da9",
    "f8d0ecdc-b7bd-44cc-b887-3d544e24e561",
    "f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c",
    "f93bfce4-e814-4ae3-9cdf-59f4dcdedf51",
    "f9d8aacd-b2a0-49f2-bd71-c2f5aadcfdd1",
    "febb430e-2d50-4f83-87a0-b5ffbb9a4943",
]

NEMO_TEST_PIDS = [
    "00a824c0-e060-495f-9ebc-79c82fef4c67",
    "00a96dee-1e8b-44cc-9cc3-aca704d2b594",
    "0228bcfd-632e-49bd-acd4-c334cf9213e9",
    "0393f34c-a2bd-4c01-99c9-f6b4ec6e786d",
    "03d2d8d1-a116-4763-8425-4ef7b1c1bd35",
    "04690e35-ab38-41db-982c-50cbdf8d0dd1",
    "05ccc92c-fcb0-4e92-84d3-de033890c7a8",
    "0909252c-3ad0-413f-96f5-7eff885b50aa",
    "091392a5-73f6-40f3-8552-fa917cf96deb",
    "0a0184b3-9e1a-4c36-98f4-00ae3beb8f01",
    "0ce74616-abf8-47c2-86d9-f821cd25efd3",
    "11f70c37-a546-439a-9fb7-da6b6ebfb0cb",
    "15a2a63f-739d-4eb0-ab8f-fc95b6299b68",
    "16799c7a-e395-435d-a4c4-a678007e1550",
    "176b4fe3-f570-4d9f-9e25-a5d218f75c8b",
    "1878c999-d523-474a-9d4e-8dde53d7324c",
    "19c9caea-2df8-4097-92f8-0a2bad055948",
    "1ab86a7f-578b-4a46-9c9c-df3be97abcca",
    "1bf5c05b-3654-4d5c-bac9-2b96edd12adf",
    "1ca6cd06-1ed5-45af-b73a-017d5e7cff48",
    "1d547041-230a-4af3-ba6a-7287de2bdec3",
    "1ff5ce1f-5500-444b-ad5f-70f90dd22ac3",
    "2122d807-13ab-494e-a1cc-e55cf24d3e9f",
    "2400a2fa-0335-480b-ac8d-a478171f3a55",
    "25e75514-43e1-4c07-9fea-581bd203b55c",
    "27109779-1b15-4d74-893f-08216d231307",
    "2ce1d485-ebce-41a2-a5ce-aa109d5a13a3",
    "2d2cdf86-4a0c-42d8-aed3-5b187f160013",
    "2d544e7f-c8af-46b5-aaa0-caf0ad4d63a1",
    "30ac1320-d671-46fc-87ef-875cdfc6b4eb",
    "335d689e-4d98-4532-b01b-7f7da89863c4",
    "39883ded-f5a2-4f4f-a98e-fb138eb8433e",
    "3a37fb5c-3e96-4a0e-ac63-2eeec6744588",
    "3b729602-20d5-4be8-a10e-24bde8fc3092",
    "3d3d5a5e-df26-43ee-80b6-2d72d85668a5",
    "41b9162e-df67-4321-b2e6-02ff851816c5",
    "437f43e1-9c8f-4be9-ac89-97885b0e03d0",
    "43d4f889-4b05-44df-8306-ea10f854776f",
    "4431f9fd-aaa8-4b10-905c-0c6869ea1088",
    "45c49ba2-a113-4446-9c6d-9b049c1f9f74",
    "485b50c8-71e1-4654-9a07-64395c15f5ed",
    "49a86b2e-3db4-42f2-8da8-7ebb7e482c70",
    "4d2e7e49-5b08-4914-afa6-fd311d957082",
    "507b20cf-4ab0-4f55-9a81-88b02839d127",
    "516b2043-0176-49a3-a986-132b3d02f28a",
    "5245b505-d0e7-4942-94d1-b33a084fab03",
    "532bb33e-f22f-454a-ba1f-561d1578b250",
    "53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2",
    "5458cb27-d065-4626-bcd8-1aa775e1115e",
    "54c8f5de-83d7-49b9-9b4c-67fe0f07289b",
    "57656bee-e32e-4848-b924-0f6f18cfdfb1",
    "5a2f99aa-d3f5-469d-882e-0a16ee869746",
    "5a5a2320-3b0d-485d-aba2-dc96357e00d8",
    "5bfcf68f-4e0b-46f8-a9e3-8e919d5ddd1c",
    "5e84c8e7-236f-4a64-a944-dc4a17c64f1d",
    "5e8ac11b-959a-49ab-a6a3-8a3397e1df0e",
    "6104a953-589c-4661-bed9-a81ccef7d0be",
    "63435e73-6a72-4007-b0da-03e6473e6870",
    "63a32e5c-f63a-450d-85cb-140947b67eaf",
    "6506252f-1141-4e11-8816-4db01b71dda0",
    "68f5f1e4-5e88-4aa5-8a1a-94b691adb11e",
    "6a22a5b4-79ec-4b3c-b023-1ec7b4a2b675",
    "6d3b68e0-3efd-4b03-b747-16e44118a0a9",
    "70da415f-444d-4148-ade7-a1f58a16fcf8",
    "72274871-80e5-4fb9-be1a-bb04acebb1de",
    "779fbed1-4b0e-4d7d-8882-6650690221a0",
    "7909c0aa-c074-4e19-aabf-b8167c682a5b",
    "799d899d-c398-4e81-abaf-1ef4b02d5475",
    "80c7f589-733c-4f23-abf3-ade8c79f0a3b",
    "81b1800e-f138-4337-abc9-bb0449b77852",
    "8386f396-0a69-48b5-98d4-18801066ea76",
    "84fd7fa3-6c2d-4233-b265-46a427d3d68d",
    "88e148d2-d554-42c2-9c41-cc6369f98c45",
    "88fe9ed9-9e3e-431a-8ecc-4cb2fe5bf448",
    "8ab470cc-de5a-4d65-92bd-9a25fbe6d07d",
    "8b31b4bd-003e-4816-a3bf-2df4cc3558f8",
    "8ca1a850-26ef-42be-8b28-c2e2d12f06d6",
    "8f4fd564-7e34-4113-a6e4-ae40faa64f7a",
    "920865d3-1581-4e81-bd63-ab2186024564",
    "96fe2729-b5a2-4e77-92bb-9df410144768",
    "9861fa24-d998-4ef5-9606-89002b8706fc",
    "9ac90b86-fc72-4ca0-9389-c8b3a77da437",
    "9b3ad89a-177f-4242-9a96-2fd98721e47f",
    "9bd71ac6-c455-41f3-8c56-59987a649ac4",
    "9e069684-a4be-4b70-b9e6-446309f977d4",
    "9f99bd34-40d4-4aac-ae2b-7bc7f7086ccb",
    "a4ed0599-4a72-4afb-8fe9-d0d046aec6aa",
    "a6bcfe32-69ce-414f-80bc-495d1b4906af",
    "a6f9e4d5-5f20-4a98-9118-22187f4d230b",
    "aac3b928-e99a-4039-ace1-af45d0130d82",
    "ac088ddb-9e1e-49ae-b341-46a8445cf809",
    "ae03524f-e493-4077-99b1-588244de9a98",
    "aeb06797-fddf-4ad2-b0ef-dd1b0f54d034",
    "aec2b14f-5dbc-400b-bf2e-dd13e711e2ff",
    "afe87fbb-3a17-461f-b333-e22903f1d70d",
    "b3065f45-7227-4099-9f8c-7e81e677d8cf",
    "b33e4298-a735-4781-8dea-d7203b4137fb",
    "b8f3a7c3-b700-446b-8d29-69691c0e3b1d",
    "b939cc85-6028-404a-995d-28c8405a07db",
    "b98f6b89-3de4-4295-be1c-59e465de1e32",
    "be76cfe3-b2bb-4358-956b-0fad07215972",
    "c07d13ed-e387-4457-8e33-1d16aed3fd92",
    "c1014051-d06b-4f85-9887-e7c42a94baf2",
    "c2363000-27a6-461e-940b-15f681496ed8",
    "c250d4f4-7516-4cf1-a8bd-04873ca9e21c",
    "c736dc60-0726-410d-8419-cb6ea025d5b1",
    "cab81176-36cd-4c37-9353-841e85027d36",
    "d0046384-16ea-4f69-bae9-165e8d0aeacf",
    "d06f4b61-42c1-4a8d-a2a5-c1cf8f7d0e47",
    "d1d9defc-6f73-489f-9f14-2c0b5e970b2d",
    "d213e786-4b1c-477d-a710-766d69fa1ac1",
    "d403c464-9d91-4338-b64b-81c299356972",
    "d552cffa-b662-40bd-b1e3-98d0a8face2c",
    "d595ee58-75bb-4df3-96de-d9b4f4d1f6ab",
    "d5e5311c-8beb-4f8f-b798-3e9bfa6bcdd8",
    "d6c810b2-5922-4127-af2b-4b104bc55c2b",
    "d70f3604-b3cc-41b0-967a-f6619b2586a7",
    "d8c7d3f2-f8e7-451d-bca1-7800f4ef52ed",
    "de5d704c-4a5b-4fb0-b4af-71614c510a8b",
    "df6012d0-d921-4d0a-af2a-2a91030d0f42",
    "e089d4d9-0a74-4ebc-951d-67098305e06b",
    "e5bc1f00-f4b8-4b35-a9b6-60d9cc7959a1",
    "e8901184-9f60-4309-bee5-c3c95030550a",
    "ea931b15-6214-4b80-8d88-7acbbe071bc0",
    "eb99c2c8-e614-4240-931b-169bed23e9f5",
    "ec2fbc3e-cb2b-48cb-a521-3a6ca15e244c",
    "ecde7e20-f135-47dd-8f96-adac49e4942e",
    "ece878b9-830d-4618-b801-ad0e7d8e7085",
    "eda7a3ac-f038-4603-9c68-816234e9c4eb",
    "f0c390da-d8e3-4b5f-8df7-bd2f153ed01b",
    "f72e60c3-3593-466a-afaf-91145c44fb2b",
    "f74a6b9a-b8a5-4c80-9c30-7dd4cdbb48c0",
    "f7c93877-ec05-4091-a003-e69fae0f2fa8",
    "fa539f97-a078-4fc6-9a12-237799466d9c",
    "fd555d39-b728-44aa-90d2-796b8bb58300",
]


def get_waveforms_coordinates(
    trace_indices,
    xy=None,
    extract_radius_um=EXTRACT_RADIUS_UM,
    return_complex=False,
    return_indices=False,
):
    """
    Reproduces the localisation code channel selection when extracting waveforms from raw data.
    Args:
        trace_indices: np.array (nspikes,): index of the trace of the detected spike )
        xy: (optional)
        extract_radius_um: radius from peak trace: all traces within this radius will be included
        return_complex: if True, returns the complex coordinates, otherwise returns a 3D x, y, z array
        return_indices: if True, returns the indices of the channels within the radius
    Returns: (np.array, np.array) (nspikes, ntraces, n_coordinates) of axial and transverse coordinates, (nspikes, ntraces) of indices
    """
    if xy is None:
        th = neuropixel.trace_header(version=1)
        xy = th["x"] + 1j * th["y"]
    channel_lookups = _get_channel_distances_indices(
        xy, extract_radius_um=extract_radius_um
    )
    inds = channel_lookups[trace_indices.astype(np.int32)]
    # add a dummy channel to have nans in the coordinates
    inds[np.isnan(inds)] = xy.size - 1
    wxy = np.r_[xy, np.nan][inds.astype(np.int32)]
    if not return_complex:
        wxy = np.stack(
            (np.real(wxy), np.imag(wxy), np.zeros_like(np.imag(wxy))), axis=2
        )
    if return_indices:
        return wxy, inds.astype(int)
    else:
        return wxy


def _get_channel_distances_indices(xy, extract_radius_um=EXTRACT_RADIUS_UM):
    """
    params: xy: ntr complex array of x and y coordinates of each channel relative to the probe
    Computes the distance between each channel and all the other channels, and find the
    indices of the channels that are within the radius.
    For each row the indices of the channels within the radius are returned.
    returns: channel_dist: ntr x ntr_wav matrix of channel indices within the radius., where ntr_wav is the
    """
    ntr = xy.shape[0]
    channel_dist = np.zeros((ntr, ntr)) * np.nan
    for i in np.arange(ntr):
        cind = np.where(np.abs(xy[i] - xy) <= extract_radius_um)[0]
        channel_dist[i, : cind.size] = cind
    # prune the matrix: only so many channels are within the radius
    channel_dist = channel_dist[:, ~np.all(np.isnan(channel_dist), axis=0)]
    return channel_dist


def save_model(path_model, classifier, meta, subfolder="", identifier=None):
    """
    Save model to disk in ubj format with associated meta-data and a hash
    The model is a set of files in a folder named after the meta-data
     'VINTAGE' and 'REGION_MAP' fields, with the hash as suffix e.g. 2023_W41_Cosmos_dfd731f0
    :param classifier:
    :param meta:
    :param path_model:
    :param identifier: optional identifier for the model, defaults to a 8 character hexdigest of the meta data
    :param subfolder: optional level to add to the model path, for example 'FOLD01' will write to
        2023_W41_Cosmos_dfd731f0/FOLD01/
    :return:
    """
    meta['MODEL_CLASS'] = (
        f"{classifier.__class__.__module__}.{classifier.__class__.__name__}"
    )
    if identifier is None:
        identifier = hashlib.md5(yaml.dump(meta).encode("utf-8")).hexdigest()[:8]
    path_model = path_model.joinpath(
        f"{meta['VINTAGE']}_{meta['REGION_MAP']}_{identifier}", subfolder
    )
    path_model.mkdir(exist_ok=True, parents=True)
    with open(path_model.joinpath("meta.yaml"), "w+") as fid:
        fid.write(yaml.dump(dict(meta)))
    classifier.save_model(path_model.joinpath("model.ubj"))
    return path_model


def download_model(local_path: Path, model_name: str, one: ONE, overwrite=False) -> Path:
    """
    download_model(Path('/mnt/s0/ephys-atlas-decoding/models'), '2024_W50_Cosmos_lid-basket-sense', one=one)
    :param local_path:
    :param model_name:
    :param one:
    :param overwrite:
    :return:
    """
    local_path = Path(local_path)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_folder(
        f"aggregates/atlas/models/{model_name}",
        local_path.joinpath(model_name),
        s3=s3,
        bucket_name=bucket_name,
        overwrite=overwrite,
    )
    return local_path.joinpath(model_name)


def atlas_pids_autism(one):
    """
    Get autism data from JP
    fmr1 mouse line
    """
    project = "angelaki_mouseASD"
    # Get all insertions for this project
    str_query = (
        f"session__projects__name__icontains,{project},"
        "session__qc__lt,50,"
        "~json__qc,CRITICAL"
    )
    insertions = one.alyx.rest("insertions", "list", django=str_query)
    # Restrict to only those with subject starting with FMR
    ins_keep = [
        item for item in insertions if item["session_info"]["subject"][0:3] == "FMR"
    ]
    return [item["id"] for item in ins_keep], ins_keep


def atlas_pids(one, tracing=False):
    django_strg = [
        "session__projects__name__icontains,ibl_neuropixel_brainwide_01",
        "session__qc__lt,50",
        "~json__qc,CRITICAL",
        # 'session__extended_qc__behavior,1',
        "session__json__IS_MOCK,False",
    ]
    if tracing:
        django_strg.append("json__extended_qc__tracing_exists,True")

    insertions = one.alyx.rest("insertions", "list", django=django_strg)
    return [item["id"] for item in insertions], insertions


def load_voltage_features(local_path, regions=None, mapping="Cosmos", dropna=True):
    """
    Load the voltage features, drop  NaNs and merge the channel information at the Allen, Beryl and Cosmos levels
    :param local_path: full path to folder containing the features table "/data/ephys-atlas/2023_W34"
    :param regions:
    :param mapping: Level of hierarchy, Cosmos, Beryl or Allen
    :return:
    """
    list_mapping = ["Cosmos", "Beryl", "Allen"]
    if mapping not in list_mapping:
        raise ValueError(f"mapping should be in {list_mapping}")

    regions = BrainRegions() if regions is None else regions
    if local_path is None:
        ## data loading section
        config = get_config()
        # this path contains channels.pqt, clusters.pqt and raw_ephys_features.pqt
        local_path = Path(config["paths"]["features"]).joinpath("latest")
    df_voltage, df_clusters, df_channels, df_probes = load_tables(Path(local_path))
    df_voltage = pd.merge(df_voltage, df_channels, left_index=True, right_index=True)
    df_voltage = df_voltage.rename(
        columns={"atlas_id": "Allen_id", "acronym": "Allen_acronym"}
    )
    df_voltage = prep_voltage_dataframe(df_voltage, mapping=mapping, regions=regions)
    df_voltage["pids"] = df_voltage.index.get_level_values(0)
    _logger.info(f"Loaded {df_voltage.shape[0]} channels")
    if dropna:
        df_voltage = df_voltage.dropna()
    _logger.info(f"Remains {df_voltage.shape[0]} channels after NaNs filtering")

    return df_voltage, df_clusters, df_channels, df_probes


def prep_voltage_dataframe(df_voltage, mapping='Allen', regions=None):
    regions = BrainRegions() if regions is None else regions
    df_voltage.replace([np.inf, -np.inf], np.nan, inplace=True)
    if mapping != "Allen":
        df_voltage[mapping + "_id"] = regions.remap(
            df_voltage["Allen_id"], source_map="Allen", target_map=mapping
        )
        df_voltage[mapping + "_acronym"] = regions.id2acronym(
            df_voltage[mapping + "_id"]
        )
    return df_voltage



def load_tables(local_path, verify=False):
    """
    :param local_path: path to the folder containing the tables
    """
    local_path = Path(local_path)
    local_path.mkdir(exist_ok=True)  # no parent here
    if not (file_raw_features := local_path.joinpath("raw_ephys_features_denoised.pqt")).exists():
        file_raw_features = local_path.joinpath("raw_ephys_features.pqt")
    df_channels = pd.read_parquet(local_path.joinpath("channels.pqt"))
    df_voltage = pd.read_parquet(file_raw_features)
    df_probes = pd.read_parquet(local_path.joinpath("probes.pqt"))
    if local_path.joinpath("clusters.pqt").exists():
        df_clusters = pd.read_parquet(local_path.joinpath("clusters.pqt"))
    else:
        df_clusters = None
    if verify:
        verify_tables(df_voltage, df_clusters, df_channels)
    return df_voltage, df_clusters, df_channels, df_probes


def read_correlogram(file_correlogram, nclusters):
    nbins = int(Path(file_correlogram).stat().st_size / nclusters / 4)
    mmap_correlogram = np.memmap(
        file_correlogram, dtype="int32", shape=(nclusters, nbins)
    )
    return mmap_correlogram


def download_tables(
    local_path, label="latest", one=None, verify=False, overwrite=False, extended=False
):
    """
    :param local_path: pathlib.Path() where the data will be stored locally
    :param label: revision string "2024_W04"
    :param one:
    :param verify: checks the indices and consistency of the dataframes and raise an error if not consistent
    :param overwrite: force redownloading if file exists
    :param extended: if True, will download also extended datasets, such as cross-correlograms that take up
    more space than just the tables (coople hundreds Mb for the table, several GB with extended data)
    :return:
    """
    # The AWS private credentials are stored in Alyx, so that only one authentication is required
    local_path = Path(local_path).joinpath(label)
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    local_files = aws.s3_download_folder(
        f"aggregates/atlas/{label}",
        local_path,
        s3=s3,
        bucket_name=bucket_name,
        overwrite=overwrite,
    )
    if extended:
        local_files = aws.s3_download_folder(
            f"aggregates/atlas/{label}_extended",
            local_path,
            s3=s3,
            bucket_name=bucket_name,
            overwrite=overwrite,
        )
    assert len(local_files), f"aggregates/atlas/{label} not found on AWS"
    return load_tables(local_path=local_path, verify=verify)


def verify_tables(df_voltage, df_clusters, df_channels):
    """
    Verify that the tables have the correct format and indices
    :param df_clusters:
    :param df_channels:
    :param df_voltage:
    :return:
    """
    assert df_clusters.index.names == ["pid", "cluster_id"]
    assert df_channels.index.names == ["pid", "channel"]
    assert df_voltage.index.names == ["pid", "channel"]


def compute_depth_dataframe(df_raw_features, df_clusters, df_channels):
    """
    Compute a features dataframe for each pid and depth along the probe,
    merging the raw voltage features, and the clusters features
    :param df_voltage:
    :param df_clusters:
    :param df_channels:
    :return:
    """
    df_depth_clusters = df_clusters.groupby(["pid", "axial_um"]).agg(
        amp_max=pd.NamedAgg(column="amp_max", aggfunc="mean"),
        amp_min=pd.NamedAgg(column="amp_min", aggfunc="mean"),
        amp_median=pd.NamedAgg(column="amp_median", aggfunc="mean"),
        amp_std_dB=pd.NamedAgg(column="amp_std_dB", aggfunc="mean"),
        contamination=pd.NamedAgg(column="contamination", aggfunc="mean"),
        contamination_alt=pd.NamedAgg(column="contamination_alt", aggfunc="mean"),
        drift=pd.NamedAgg(column="drift", aggfunc="mean"),
        missed_spikes_est=pd.NamedAgg(column="missed_spikes_est", aggfunc="mean"),
        noise_cutoff=pd.NamedAgg(column="noise_cutoff", aggfunc="mean"),
        presence_ratio=pd.NamedAgg(column="presence_ratio", aggfunc="mean"),
        presence_ratio_std=pd.NamedAgg(column="presence_ratio_std", aggfunc="mean"),
        slidingRP_viol=pd.NamedAgg(column="slidingRP_viol", aggfunc="mean"),
        spike_count=pd.NamedAgg(column="spike_count", aggfunc="mean"),
        firing_rate=pd.NamedAgg(column="firing_rate", aggfunc="mean"),
        label=pd.NamedAgg(column="label", aggfunc="mean"),
        x=pd.NamedAgg(column="x", aggfunc="mean"),
        y=pd.NamedAgg(column="y", aggfunc="mean"),
        z=pd.NamedAgg(column="z", aggfunc="mean"),
        acronym=pd.NamedAgg(column="acronym", aggfunc="first"),
        atlas_id=pd.NamedAgg(column="atlas_id", aggfunc="first"),
    )

    df_voltage = df_raw_features.merge(df_channels, left_index=True, right_index=True)
    df_depth_raw = df_voltage.groupby(["pid", "axial_um"]).agg(
        alpha_mean=pd.NamedAgg(column="alpha_mean", aggfunc="mean"),
        alpha_std=pd.NamedAgg(column="alpha_std", aggfunc="mean"),
        spike_count=pd.NamedAgg(column="spike_count", aggfunc="mean"),
        cloud_x_std=pd.NamedAgg(column="cloud_x_std", aggfunc="mean"),
        cloud_y_std=pd.NamedAgg(column="cloud_y_std", aggfunc="mean"),
        cloud_z_std=pd.NamedAgg(column="cloud_z_std", aggfunc="mean"),
        peak_trace_idx=pd.NamedAgg(column="peak_trace_idx", aggfunc="mean"),
        peak_time_idx=pd.NamedAgg(column="peak_time_idx", aggfunc="mean"),
        peak_val=pd.NamedAgg(column="peak_val", aggfunc="mean"),
        trough_time_idx=pd.NamedAgg(column="trough_time_idx", aggfunc="mean"),
        trough_val=pd.NamedAgg(column="trough_val", aggfunc="mean"),
        tip_time_idx=pd.NamedAgg(column="tip_time_idx", aggfunc="mean"),
        tip_val=pd.NamedAgg(column="tip_val", aggfunc="mean"),
        rms_ap=pd.NamedAgg(column="rms_ap", aggfunc="mean"),
        rms_lf=pd.NamedAgg(column="rms_lf", aggfunc="mean"),
        psd_delta=pd.NamedAgg(column="psd_delta", aggfunc="mean"),
        psd_theta=pd.NamedAgg(column="psd_theta", aggfunc="mean"),
        psd_alpha=pd.NamedAgg(column="psd_alpha", aggfunc="mean"),
        psd_beta=pd.NamedAgg(column="psd_beta", aggfunc="mean"),
        psd_gamma=pd.NamedAgg(column="psd_gamma", aggfunc="mean"),
        x=pd.NamedAgg(column="x", aggfunc="mean"),
        y=pd.NamedAgg(column="y", aggfunc="mean"),
        z=pd.NamedAgg(column="z", aggfunc="mean"),
        acronym=pd.NamedAgg(column="acronym", aggfunc="first"),
        atlas_id=pd.NamedAgg(column="atlas_id", aggfunc="first"),
        histology=pd.NamedAgg(column="histology", aggfunc="first"),
    )
    df_depth = df_depth_raw.merge(df_depth_clusters, left_index=True, right_index=True)
    return df_depth

def get_config():
    file_yaml = Path(__file__).parents[2].joinpath("config-ephys-atlas.yaml")
    with open(file_yaml, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def compute_summary_stat(df_voltage, features):
    """
    Summary statistics
    :param df_voltage:
    :param features:
    :return:
    """
    # The behavior of loc is inconsistent
    # If you input a str instead of a list, it returns a Series instead of a dataframe
    if type(features) != list:  # Make sure input is a list
        features = [features]

    summary = (
        df_voltage.loc[:, features]
        .agg(["median", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
        .T
    )
    summary.columns = ["median", "q05", "q95"]
    summary["dq"] = summary["q95"] - summary["q05"]
    return summary


def sort_feature(values, features, ascending=True):
    """
    Sort the value (metrics being p-value, or else)
    :param values:
    :param features:
    :param ascending:
    :return:
    """
    id_sort = np.argsort(values)
    if not ascending:
        id_sort = np.flip(id_sort)
    features_sort = features[id_sort]
    values_sort = values[id_sort]
    return values_sort, features_sort


def prepare_mat_plot(array_in, id_feat, diag_val=0):
    """
    From the matrix storing the results of brain-to-brain regions comparison in the upper triangle for all features,
    select a feature and create a matrix with transpose for plotting in 2D
    :param array_in: array of N region x N region x N feature
    :param id_feat: index of feature that will be displayed
    :return:
    """
    mat_plot = np.squeeze(array_in[:, :, id_feat].copy())
    mat_plot[np.tril_indices_from(mat_plot)] = diag_val  # replace Nan by 0
    mat_plot = mat_plot + mat_plot.T  # add transpose for display
    return mat_plot


def prepare_df_voltage(df_voltage, df_channels, br=None):
    if br is None:
        br = BrainRegions()
    df_voltage = pd.merge(
        df_voltage, df_channels, left_index=True, right_index=True
    ).dropna()
    df_voltage["cosmos_id"] = br.remap(
        df_voltage["atlas_id"], source_map="Allen", target_map="Cosmos"
    )
    df_voltage["beryl_id"] = br.remap(
        df_voltage["atlas_id"], source_map="Allen", target_map="Beryl"
    )

    df_voltage = df_voltage.loc[
        ~df_voltage["cosmos_id"].isin(br.acronym2id(["void", "root"]))
    ]
    for feat in ["rms_ap", "rms_lf"]:
        df_voltage[feat] = 20 * np.log10(df_voltage[feat])
    df_voltage["spike_count_log"] = np.log10(df_voltage["spike_count"] + 1)

    # Add in peak_to_trough_ratio + peak_to_trough_duration
    df_voltage = peak_to_trough_ratio(df_voltage)
    df_voltage["peak_to_trough_duration"] = (
        df_voltage["trough_time_secs"] - df_voltage["peak_time_secs"]
    )
    return df_voltage
