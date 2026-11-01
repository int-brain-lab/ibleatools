"""Quality-control filter of the Ephys Atlas insertions -- GENERATED, do not edit by hand.

Built by ``scripts/build_pid_qc_filter.py`` (2026-11-01, ibleatools a495d45) from feature release
2026_W42, at level L3 of ``ephysatlas.qc.levels``: 1049 insertions, 76 with at least one bad
component (21 alignment, 25 lf only, 28 ap/waveform, 2 several bands), 24 to review.

An insertion is bad at L3 when it fails one test beyond 4.5 SD, or a test without a limit in SDs
(its tracks swapped with the session's other probe, a fifth of its channels dead or noisy). The
tests are ``ephysatlas.qc.levels.TESTS`` and ``FIXED_TESTS``: each compares the probe's recording
with what the other probes record in the same brain regions; no model, auto-alignment or manual
label decides. Each component -- ``alignment``, ``lf``, ``ap``, ``waveform`` -- is ``bad`` when the
tests failed at L3 point to it, ``review`` when it is bad only at the next, more sensitive level L4
(one test beyond 4 SD), else ``good``. Every insertion that is not good on all four is listed with
its labels and the tests it fails (at L3; for an insertion only to review, at L4). Unlisted
insertions are good.

Use :func:`excluded_pids` for the insertions to leave out of a dataset, e.g. ``excluded_pids()``
(bad on any component) for the channel-level model, or ``excluded_pids(("alignment", "ap",
"waveform"))`` for the unit-level model, and pass them as ``exclude_pids`` to the data loaders
(``spatial_encoder.utils.LoadInsertionData``, ``alignment.ChannelDataset.load``,
``unit_level_encoder.prepare_data.prepare_latest_cells_encoder_data``), which use exactly these by
default.
"""

VINTAGE = "2026_W42"
LEVEL = "L3"
COMPONENTS = ("alignment", "lf", "ap", "waveform")

# pid: (alignment, lf, ap, waveform, tests failed)
pid_qc_filter = {
    "03fc9668-046c-445f-953a-ee9258520ea0": ("bad", "good", "good", "good", "swap"),
    "04db6b9e-a80c-4507-a98e-ad76294ac444": ("good", "good", "good", "bad", "wf_shape"),
    "0a85a28b-e9e3-44ec-ad4d-88ba9116ff63": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "0c15a331-09ac-445c-837f-6afb5e377e56": (
        "good",
        "good",
        "good",
        "review",
        "wf_level",
    ),
    "0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b": ("good", "bad", "good", "good", "lf_level"),
    "0fed7207-f747-428b-b4c0-854cabb50d9e": ("good", "bad", "good", "good", "lf_level"),
    "100433fa-2c59-4432-8295-aa27657fe3fb": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "16ad5eef-3fa6-4c75-9296-29bf40c5cfaa": ("good", "good", "bad", "bad", "channels"),
    "176b4fe3-f570-4d9f-9e25-a5d218f75c8b": ("good", "bad", "good", "good", "lf_shape"),
    "19e0bf1e-018e-40cb-9acd-af8f0f66bd4a": ("good", "good", "good", "bad", "wf_shape"),
    "1dd218c9-ac97-4d91-80d0-a8a660bf7395": (
        "review",
        "good",
        "good",
        "good",
        "sideways",
    ),
    "1fa9efcf-def3-40b1-86ae-ca411da98ef2": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "24572b57-a9c0-463b-8ef9-d232760d9792": ("good", "good", "bad", "bad", "ap_level"),
    "25e75514-43e1-4c07-9fea-581bd203b55c": (
        "good",
        "review",
        "good",
        "good",
        "lf_shape",
    ),
    "2781c08e-a356-4a0d-8c28-4f94d1c87f69": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "27a0977e-8d53-40b2-a9e1-77d8523a0ad3": ("good", "bad", "good", "good", "lf_shape"),
    "29418b18-4db6-436f-a85e-b0f1d38d2e62": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "2dd99c91-292f-44e3-bbf2-8cfa56015106": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "316a733a-5358-4d1d-9f7f-179ba3c90adf": ("good", "good", "bad", "bad", "channels"),
    "3270d18e-df94-42d9-bace-6e9ee95c29c7": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "379105fd-dbb1-40d9-8a51-ae5a8ef8df04": (
        "bad",
        "good",
        "review",
        "review",
        "depth",
    ),
    "3cf66b31-9f69-4050-93d8-c0de1b7ce944": ("good", "good", "bad", "bad", "ap_level"),
    "3d12814e-594a-4685-9f7c-5a2f0a53b686": (
        "good",
        "good",
        "review",
        "review",
        "ap_shape",
    ),
    "43436b4b-0431-407f-b83d-d657ec22b5c6": ("good", "bad", "good", "good", "lf_level"),
    "47da98a8-f282-4830-92c2-af0e1d4f00e2": ("good", "bad", "good", "good", "lf_level"),
    "49c2ea3d-2b50-4e8a-b124-9e190960784e": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "4d2e7e49-5b08-4914-afa6-fd311d957082": (
        "good",
        "bad",
        "bad",
        "bad",
        "lf_level;ap_level;wf_level",
    ),
    "4e6dfe08-cab0-4a05-903b-94283cb9f8e7": (
        "good",
        "review",
        "good",
        "good",
        "lf_shape",
    ),
    "4f46198d-c54d-4627-9e4e-3c8fd4c3c070": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "50cdbc2b-4ecd-4044-955d-9654505f7f98": ("bad", "good", "good", "good", "depth"),
    "50f1512d-dd41-4a0c-b3ab-b0564f0424d7": ("review", "good", "good", "good", "depth"),
    "511afaa5-fdc4-4166-b4c0-4629ec5e652e": ("good", "good", "bad", "bad", "channels"),
    "55423355-281c-41f2-8ef7-5aa098ffbbed": ("review", "good", "good", "good", "depth"),
    "557c2d77-640e-4238-be04-c1ac07d8eab4": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "5810514e-2a86-4a34-b7bd-1e4b0b601295": ("bad", "good", "good", "good", "depth"),
    "5bc219f4-06a0-4f3e-a9e9-304d8c36797b": ("good", "bad", "good", "good", "lf_level"),
    "5c63d860-1e3c-481b-a290-9f299a5421f5": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;wf_level",
    ),
    "5eebb5d6-cf9b-49a0-b297-0701c4d90173": ("bad", "good", "good", "good", "depth"),
    "697b57b7-e332-455b-b958-90e4e1f2b43f": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "697d9e03-edc6-430a-ae76-94628c510161": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "6a098711-5423-4072-8909-7cff0e2d4531": ("good", "good", "bad", "bad", "ap_level"),
    "6e61f777-90b9-4656-af30-9ed54d426c5a": ("good", "good", "bad", "bad", "ap_shape"),
    "759860bc-732d-4b9e-aeb4-154ef37c056b": ("good", "bad", "good", "good", "lf_level"),
    "75d5e438-b08c-47d3-8ac0-992ec63c9c1b": ("good", "good", "bad", "bad", "ap_level"),
    "7649b822-e9ed-409c-b325-05235fede18b": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "76de0e1a-30aa-4713-9fe5-25ad2dff653f": (
        "review",
        "good",
        "good",
        "review",
        "wf_level;depth",
    ),
    "76f16631-cbf7-4883-9437-4e8d8c4f34a3": ("good", "good", "bad", "bad", "ap_shape"),
    "7a02c052-ce65-4d60-99d7-4723cdb334a3": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "80624507-4be6-4689-92df-0e2c26c3faf3": ("bad", "good", "good", "good", "depth"),
    "810416d1-ce3c-4b2e-a827-6cdc65c8cd5c": ("good", "bad", "good", "good", "lf_level"),
    "81f0087b-2bd1-4e48-8e86-e8206aee3d9d": ("bad", "good", "good", "good", "depth"),
    "88fe9ed9-9e3e-431a-8ecc-4cb2fe5bf448": ("review", "good", "good", "good", "depth"),
    "89606895-287e-4559-8536-9830b047af34": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "8aba5f8c-4438-4588-9bb0-b20d2fd3eca5": ("good", "bad", "good", "good", "lf_level"),
    "8e40a3f7-d228-4e1a-8470-9a04fcda59be": ("bad", "good", "good", "good", "depth"),
    "905ed797-ae02-4a74-8267-1bd6ac0aa589": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "9486a103-c53e-4a60-b202-4efcc6278d44": (
        "good",
        "review",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "971648ea-a685-411c-9e5f-2ceab3b2fcec": ("good", "bad", "good", "good", "lf_level"),
    "97415ee8-0b90-456c-83c2-05e7a398c06c": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;wf_level",
    ),
    "9ac90b86-fc72-4ca0-9389-c8b3a77da437": ("bad", "good", "good", "good", "depth"),
    "9bd71ac6-c455-41f3-8c56-59987a649ac4": ("bad", "good", "good", "good", "swap"),
    "a226c68d-55d8-4b53-9596-7c88a3ffa259": (
        "review",
        "review",
        "good",
        "good",
        "lf_shape;sideways",
    ),
    "a3ebc7f4-813a-4f06-a33e-9825c878f9c2": ("good", "good", "bad", "bad", "ap_shape"),
    "a85b9795-f99c-4c1d-a376-8b5ef095ffd7": ("good", "good", "bad", "bad", "ap_level"),
    "aac3b928-e99a-4039-ace1-af45d0130d82": (
        "review",
        "good",
        "good",
        "good",
        "sideways",
    ),
    "ad5277e5-933d-45e4-bade-c9bd0e1f144a": (
        "bad",
        "bad",
        "bad",
        "bad",
        "lf_level;ap_level;ap_shape;swap",
    ),
    "ad597f5f-5201-4e02-9a28-1fb1a75746cc": (
        "bad",
        "bad",
        "good",
        "good",
        "lf_level;sideways",
    ),
    "af775340-53d0-43cd-9cda-48e32b95adca": ("good", "bad", "good", "good", "lf_shape"),
    "b3065f45-7227-4099-9f8c-7e81e677d8cf": ("good", "good", "bad", "bad", "ap_shape"),
    "babba401-64eb-4861-a144-c60f68a04ae6": ("good", "bad", "good", "good", "lf_shape"),
    "bae9adba-8064-4f42-a7c5-57173c6f4c51": ("good", "good", "bad", "bad", "ap_level"),
    "bef05a5c-68c3-4513-87c7-b3151c88da8e": (
        "bad",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape;sideways",
    ),
    "bfc32fc4-9077-4806-a544-94291f8217de": ("bad", "good", "good", "good", "depth"),
    "c3ef78d5-7c75-48a4-96b7-f8ecc91c2eca": ("review", "good", "good", "good", "depth"),
    "c4f6665f-8be5-476b-a6e8-d81eeae9279d": (
        "good",
        "review",
        "good",
        "good",
        "lf_shape",
    ),
    "c64201ea-262e-45fe-8f41-5a074780cd59": (
        "good",
        "good",
        "good",
        "review",
        "wf_shape",
    ),
    "c9664185-d3fd-4e0e-89cf-77c402038938": ("good", "good", "good", "bad", "wf_shape"),
    "c9669829-36cb-400d-bd77-01c56e3f5378": (
        "bad",
        "bad",
        "bad",
        "bad",
        "lf_level;lf_shape;ap_level;depth;sideways",
    ),
    "ca073754-be17-43b7-a38a-0c1e5563ff32": ("review", "good", "good", "good", "depth"),
    "d3507787-60c8-4cea-a61e-0c708b4361d8": (
        "bad",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape;wf_level;wf_shape;sideways",
    ),
    "d8c7d3f2-f8e7-451d-bca1-7800f4ef52ed": ("good", "bad", "good", "good", "lf_level"),
    "da17779f-7728-4c51-8b54-b88764c0908d": ("bad", "good", "good", "good", "depth"),
    "dae7e5b4-7792-4792-8bbb-d5c0fd6a5c5e": ("good", "bad", "good", "good", "lf_level"),
    "db078408-671a-4dae-ab58-e231ecaafc71": (
        "bad",
        "bad",
        "good",
        "good",
        "lf_level;sideways",
    ),
    "db1965cb-e836-4b87-8c42-ee6abf0b784d": ("good", "bad", "good", "good", "lf_level"),
    "dc50c3de-5d84-4408-9725-22ae55b93522": ("good", "good", "bad", "bad", "ap_level"),
    "dd643cc7-3f64-4986-bfc8-52f00d110cca": (
        "good",
        "good",
        "good",
        "review",
        "wf_shape",
    ),
    "e22d047c-5beb-4e5b-9a3d-53b637743856": (
        "bad",
        "good",
        "good",
        "good",
        "sideways;swap",
    ),
    "e6402305-5028-42aa-975b-c540c882b131": ("good", "good", "bad", "bad", "ap_level"),
    "ecd07b7e-6450-4e31-bef1-f195129eb3d3": (
        "bad",
        "bad",
        "good",
        "good",
        "lf_level;sideways",
    ),
    "edc82c06-83fd-44be-a41e-52fdeea728b4": (
        "bad",
        "bad",
        "good",
        "bad",
        "lf_level;wf_shape;depth",
    ),
    "f72e60c3-3593-466a-afaf-91145c44fb2b": ("good", "good", "good", "bad", "wf_shape"),
    "f77a3e90-df1f-4c25-b423-fb2f8a0bcbc9": ("good", "good", "bad", "bad", "ap_shape"),
    "f93bfce4-e814-4ae3-9cdf-59f4dcdedf51": ("good", "good", "bad", "bad", "channels"),
    "fb0de39b-c750-4adc-af18-12bf8a22ed41": ("good", "bad", "good", "good", "lf_level"),
    "fb2f994b-5be8-4c1c-81dd-aad4b19ddcf7": (
        "good",
        "bad",
        "good",
        "bad",
        "lf_level;lf_shape;wf_level",
    ),
    "fcc167d6-cb53-4dc7-a58f-06a53cfabc84": (
        "good",
        "good",
        "review",
        "review",
        "ap_level",
    ),
    "fcd89a5b-30a1-4829-8f51-a72b73534141": ("good", "good", "bad", "bad", "ap_level"),
    "fdeff6df-7216-4b83-ab64-241e9fd173da": ("good", "bad", "good", "good", "lf_level"),
    "fe9c2533-d7cd-48d9-b208-837282961f2e": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
}


def qc_labels(pid: str) -> dict:
    """``{component: label}`` of an insertion (all ``good`` when unlisted) and ``checks``."""
    entry = pid_qc_filter.get(str(pid))
    if entry is None:
        return {**{c: "good" for c in COMPONENTS}, "checks": ""}
    return {**dict(zip(COMPONENTS, entry[:4])), "checks": entry[4]}


def excluded_pids(components=COMPONENTS, include_review: bool = False) -> list:
    """Insertions bad (or, with ``include_review``, also to review) on any of ``components``."""
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(
            f"unknown components {sorted(unknown)}; choose from {COMPONENTS}"
        )
    drop = ("bad", "review") if include_review else ("bad",)
    idx = [COMPONENTS.index(c) for c in components]
    return sorted(p for p, e in pid_qc_filter.items() if any(e[i] in drop for i in idx))
