"""Quality-control filter of the Ephys Atlas insertions -- GENERATED, do not edit by hand.

Built by ``scripts/build_pid_qc_filter.py`` (2026-11-01, ibleatools 7f066e9) from feature release
2026_W42, at level L3 of ``ephysatlas.qc.levels``: 1049 insertions, 93 with at least one bad
component (50 alignment, 22 lf only, 21 ap/waveform), 37 to review.

An insertion is bad at L3 when it fails a band test beyond 5 SD, a position test beyond 3.5 SD, or
a track rank beyond 0.65, or a test with a natural limit (it and another probe of the mouse fitting
better swapped, a fifth of its channels dead or noisy). The tests are
``ephysatlas.qc.levels.TESTS``, ``TRACK_TEST`` and ``FIXED_TESTS``: each compares the probe's
recording with what the other probes record in the same brain regions; no model, auto-alignment or
manual label decides. Each component -- ``alignment``, ``lf``, ``ap``, ``waveform`` -- is ``bad``
when the tests failed at L3 point to it, ``review`` when it is bad only at the next, more sensitive
level L4 (a band test beyond 4.5 SD, a position test beyond 3 SD, or a track rank beyond 0.55),
else ``good``. Every insertion that is not good on all four is listed with its labels and the tests
it fails (at L3; for an insertion only to review, at L4). Unlisted insertions are good.

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
    "0228bcfd-632e-49bd-acd4-c334cf9213e9": ("review", "good", "good", "good", "track"),
    "03fc9668-046c-445f-953a-ee9258520ea0": (
        "bad",
        "good",
        "good",
        "good",
        "track;swap",
    ),
    "04db6b9e-a80c-4507-a98e-ad76294ac444": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track",
    ),
    "0685d767-80a2-49b6-a51a-0e9b1dd7a14d": ("review", "good", "good", "good", "track"),
    "0925fb1b-cf83-4f55-bfb7-aa52f993a404": (
        "bad",
        "good",
        "bad",
        "bad",
        "ap_level;track",
    ),
    "0c9ee229-2a3d-45ae-b277-87e9da0cf41f": ("review", "good", "good", "good", "track"),
    "0ece5c6a-7d1e-4365-893d-ac1cc04f1d7b": ("good", "bad", "good", "good", "lf_level"),
    "0fed7207-f747-428b-b4c0-854cabb50d9e": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "11f70c37-a546-439a-9fb7-da6b6ebfb0cb": ("bad", "good", "good", "good", "track"),
    "16ad5eef-3fa6-4c75-9296-29bf40c5cfaa": ("good", "good", "bad", "bad", "channels"),
    "176b4fe3-f570-4d9f-9e25-a5d218f75c8b": ("good", "bad", "good", "good", "lf_shape"),
    "19e0bf1e-018e-40cb-9acd-af8f0f66bd4a": (
        "good",
        "good",
        "good",
        "review",
        "wf_shape",
    ),
    "1bf5c05b-3654-4d5c-bac9-2b96edd12adf": ("review", "good", "good", "good", "track"),
    "1fde7d42-dbf1-4a27-9140-9758abaa50af": (
        "bad",
        "good",
        "good",
        "good",
        "track;swap",
    ),
    "20469b30-1cd8-4078-9883-a840a0f6d787": ("bad", "good", "good", "good", "track"),
    "2201eb05-aebe-4bda-905e-b5cc1baf3840": ("review", "good", "good", "good", "track"),
    "2400a2fa-0335-480b-ac8d-a478171f3a55": ("review", "good", "good", "good", "track"),
    "24572b57-a9c0-463b-8ef9-d232760d9792": (
        "good",
        "good",
        "review",
        "review",
        "ap_level",
    ),
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
    "27a0977e-8d53-40b2-a9e1-77d8523a0ad3": ("bad", "review", "good", "good", "track"),
    "29418b18-4db6-436f-a85e-b0f1d38d2e62": ("good", "bad", "good", "good", "lf_level"),
    "2dd99c91-292f-44e3-bbf2-8cfa56015106": ("good", "bad", "good", "good", "lf_level"),
    "316a733a-5358-4d1d-9f7f-179ba3c90adf": ("good", "good", "bad", "bad", "channels"),
    "3270d18e-df94-42d9-bace-6e9ee95c29c7": ("good", "good", "bad", "bad", "ap_shape"),
    "37137b1d-34d4-4183-9796-abfc9ffb6abe": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track",
    ),
    "38852fa3-907f-4d41-83d8-a8d8a8de7aef": ("bad", "good", "good", "good", "swap"),
    "3a37fb5c-3e96-4a0e-ac63-2eeec6744588": ("review", "good", "good", "good", "track"),
    "3cf66b31-9f69-4050-93d8-c0de1b7ce944": ("good", "good", "bad", "bad", "ap_level"),
    "3d12814e-594a-4685-9f7c-5a2f0a53b686": (
        "bad",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape;track;swap",
    ),
    "43436b4b-0431-407f-b83d-d657ec22b5c6": ("good", "bad", "good", "good", "lf_level"),
    "43b9b189-5221-46a0-928a-e137bc326534": (
        "bad",
        "good",
        "good",
        "good",
        "track;swap",
    ),
    "47da98a8-f282-4830-92c2-af0e1d4f00e2": (
        "good",
        "review",
        "good",
        "good",
        "lf_level",
    ),
    "49c2ea3d-2b50-4e8a-b124-9e190960784e": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "4d2e7e49-5b08-4914-afa6-fd311d957082": (
        "good",
        "review",
        "bad",
        "bad",
        "ap_level;ap_shape;wf_level",
    ),
    "4ea45238-55b1-4d54-ba92-efa47feb9f57": ("bad", "good", "good", "good", "track"),
    "4f922a5b-5014-455d-8cd5-7caed78af615": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track",
    ),
    "50445b30-f4ef-4405-b014-1d792ddfed88": (
        "bad",
        "good",
        "good",
        "good",
        "sideways;track",
    ),
    "511afaa5-fdc4-4166-b4c0-4629ec5e652e": (
        "bad",
        "good",
        "bad",
        "bad",
        "sideways;channels",
    ),
    "55423355-281c-41f2-8ef7-5aa098ffbbed": ("bad", "good", "good", "good", "depth"),
    "57acb640-4f64-4a89-bedf-69351a3aea75": ("review", "good", "good", "good", "track"),
    "5810514e-2a86-4a34-b7bd-1e4b0b601295": ("review", "good", "good", "good", "depth"),
    "5a4740a3-8e93-495d-b6cf-83dccba01d7a": ("bad", "good", "good", "good", "track"),
    "5bc219f4-06a0-4f3e-a9e9-304d8c36797b": ("good", "bad", "good", "good", "lf_level"),
    "5c63d860-1e3c-481b-a290-9f299a5421f5": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;wf_level",
    ),
    "5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e": ("review", "good", "good", "good", "track"),
    "5d9a5895-0f28-416c-a6d6-3e781658ea57": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track",
    ),
    "64346da1-c8f0-494f-a9b9-c415d7da582b": ("review", "good", "good", "good", "depth"),
    "65ad38fa-e491-4821-9e9f-d145f052498f": ("bad", "good", "good", "good", "track"),
    "6638cfb3-3831-4fc2-9327-194b76cf22e1": ("good", "bad", "good", "good", "lf_level"),
    "688606f8-9733-46f2-8331-b2dd84c5d385": ("bad", "good", "good", "good", "track"),
    "6925a312-966c-4e41-b79e-4594da57a2ee": ("review", "good", "good", "good", "track"),
    "697d9e03-edc6-430a-ae76-94628c510161": ("good", "bad", "good", "good", "lf_level"),
    "6a098711-5423-4072-8909-7cff0e2d4531": ("good", "good", "bad", "bad", "ap_level"),
    "6e61f777-90b9-4656-af30-9ed54d426c5a": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "727ff7c4-d63f-4bd8-9888-f134a25b874f": ("review", "good", "good", "good", "track"),
    "73dadb15-9219-49fd-b6bc-4d0a1284b620": ("review", "good", "good", "good", "track"),
    "7441d0df-ef9b-419b-ae4d-b9bd7c2f17cb": (
        "bad",
        "good",
        "good",
        "good",
        "sideways;track;swap",
    ),
    "759860bc-732d-4b9e-aeb4-154ef37c056b": (
        "bad",
        "bad",
        "good",
        "good",
        "lf_level;sideways;track",
    ),
    "75d5e438-b08c-47d3-8ac0-992ec63c9c1b": ("good", "good", "bad", "bad", "ap_level"),
    "76de0e1a-30aa-4713-9fe5-25ad2dff653f": (
        "good",
        "good",
        "good",
        "review",
        "wf_level",
    ),
    "76f16631-cbf7-4883-9437-4e8d8c4f34a3": (
        "good",
        "good",
        "review",
        "review",
        "ap_shape",
    ),
    "784c5282-2749-48ca-b211-100d0b24e29b": ("bad", "good", "good", "good", "track"),
    "7a02c052-ce65-4d60-99d7-4723cdb334a3": ("good", "bad", "good", "good", "lf_level"),
    "7b468b69-3dd1-4832-a491-5d1342b6e7f7": ("bad", "good", "good", "good", "swap"),
    "7c167d19-7b70-45bf-b0e8-9035227a359e": ("bad", "good", "good", "good", "track"),
    "7eac3ccd-4c7d-4584-bf5b-8c8424cf8c14": (
        "review",
        "good",
        "good",
        "good",
        "sideways",
    ),
    "810416d1-ce3c-4b2e-a827-6cdc65c8cd5c": ("good", "bad", "good", "good", "lf_level"),
    "81f0087b-2bd1-4e48-8e86-e8206aee3d9d": ("bad", "good", "good", "good", "depth"),
    "89606895-287e-4559-8536-9830b047af34": (
        "good",
        "bad",
        "good",
        "good",
        "lf_level;lf_shape",
    ),
    "8aba5f8c-4438-4588-9bb0-b20d2fd3eca5": (
        "bad",
        "bad",
        "good",
        "good",
        "lf_level;track",
    ),
    "8e40a3f7-d228-4e1a-8470-9a04fcda59be": (
        "bad",
        "good",
        "good",
        "good",
        "depth;sideways",
    ),
    "905ed797-ae02-4a74-8267-1bd6ac0aa589": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "91fb4ed3-ee27-4d7e-ba5d-f61202ed9884": ("review", "good", "good", "good", "track"),
    "938ade7f-71fa-4117-b12d-ae8084de395c": ("review", "good", "good", "good", "track"),
    "9486a103-c53e-4a60-b202-4efcc6278d44": (
        "bad",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape;depth;track;swap",
    ),
    "971648ea-a685-411c-9e5f-2ceab3b2fcec": (
        "bad",
        "bad",
        "good",
        "good",
        "lf_level;track",
    ),
    "97415ee8-0b90-456c-83c2-05e7a398c06c": (
        "good",
        "good",
        "review",
        "review",
        "ap_level;wf_level",
    ),
    "9bca04fe-40bf-467a-8f85-b0521a0ba513": ("bad", "good", "good", "good", "track"),
    "9bd71ac6-c455-41f3-8c56-59987a649ac4": ("bad", "good", "good", "good", "swap"),
    "9e069684-a4be-4b70-b9e6-446309f977d4": ("bad", "good", "good", "good", "track"),
    "9f99bd34-40d4-4aac-ae2b-7bc7f7086ccb": (
        "review",
        "good",
        "good",
        "good",
        "sideways",
    ),
    "a226c68d-55d8-4b53-9596-7c88a3ffa259": ("review", "good", "good", "good", "track"),
    "a3ebc7f4-813a-4f06-a33e-9825c878f9c2": ("good", "good", "bad", "bad", "ap_shape"),
    "a4ed0599-4a72-4afb-8fe9-d0d046aec6aa": ("review", "good", "good", "good", "track"),
    "a85b9795-f99c-4c1d-a376-8b5ef095ffd7": (
        "good",
        "good",
        "review",
        "review",
        "ap_level",
    ),
    "aac3b928-e99a-4039-ace1-af45d0130d82": (
        "bad",
        "good",
        "good",
        "good",
        "sideways;track",
    ),
    "ad5277e5-933d-45e4-bade-c9bd0e1f144a": (
        "bad",
        "review",
        "bad",
        "bad",
        "ap_level;swap",
    ),
    "ad597f5f-5201-4e02-9a28-1fb1a75746cc": ("good", "bad", "good", "good", "lf_level"),
    "ad7d5c25-7f74-4812-91ef-060a86911421": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track",
    ),
    "af775340-53d0-43cd-9cda-48e32b95adca": (
        "good",
        "review",
        "good",
        "good",
        "lf_shape",
    ),
    "b3065f45-7227-4099-9f8c-7e81e677d8cf": ("good", "good", "bad", "bad", "ap_shape"),
    "b549e337-acc5-4837-9c56-e53b173fe5fe": ("bad", "good", "good", "good", "track"),
    "b910cbe2-0c68-40b2-b9f1-268e7c28320a": ("review", "good", "good", "good", "track"),
    "bae9adba-8064-4f42-a7c5-57173c6f4c51": ("good", "good", "bad", "bad", "ap_level"),
    "bef05a5c-68c3-4513-87c7-b3151c88da8e": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "bf57f9ed-8c55-4cbe-ac75-c1200b74ddf0": ("review", "good", "good", "good", "track"),
    "bf827e50-eed3-4984-908e-c48ffbeae7b5": ("bad", "good", "good", "good", "track"),
    "c3ef78d5-7c75-48a4-96b7-f8ecc91c2eca": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track",
    ),
    "c872ca65-8319-4901-856e-7dc507e2dd9c": (
        "bad",
        "good",
        "good",
        "good",
        "depth;sideways;track",
    ),
    "c9664185-d3fd-4e0e-89cf-77c402038938": (
        "good",
        "good",
        "good",
        "review",
        "wf_shape",
    ),
    "c9669829-36cb-400d-bd77-01c56e3f5378": (
        "bad",
        "bad",
        "bad",
        "bad",
        "lf_level;lf_shape;ap_level;track;swap",
    ),
    "cbaa4916-4912-4f55-af8b-9044f182534e": ("bad", "good", "good", "good", "track"),
    "ce0dc660-f19e-46a3-94f9-646bebae6805": (
        "bad",
        "good",
        "good",
        "good",
        "track;swap",
    ),
    "d23dec01-b35e-4950-9afa-72a1b3e74148": ("bad", "good", "good", "good", "swap"),
    "d3507787-60c8-4cea-a61e-0c708b4361d8": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape;wf_shape",
    ),
    "d42dfc14-537e-43cf-a919-9f1aad91a0ff": ("review", "good", "good", "good", "track"),
    "d461eaec-f2cf-4a57-9a8c-e3eba6387359": (
        "good",
        "good",
        "bad",
        "bad",
        "ap_level;ap_shape",
    ),
    "d8c7d3f2-f8e7-451d-bca1-7800f4ef52ed": ("good", "bad", "good", "good", "lf_level"),
    "dae7e5b4-7792-4792-8bbb-d5c0fd6a5c5e": (
        "review",
        "bad",
        "good",
        "good",
        "lf_level",
    ),
    "db078408-671a-4dae-ab58-e231ecaafc71": ("good", "bad", "good", "good", "lf_level"),
    "db1965cb-e836-4b87-8c42-ee6abf0b784d": ("good", "bad", "good", "good", "lf_level"),
    "dc50c3de-5d84-4408-9725-22ae55b93522": ("good", "good", "bad", "bad", "ap_level"),
    "e033ef96-3590-4e52-bb01-6c6d75bab083": ("review", "good", "good", "good", "track"),
    "e22d047c-5beb-4e5b-9a3d-53b637743856": ("bad", "good", "good", "good", "swap"),
    "e6402305-5028-42aa-975b-c540c882b131": ("good", "good", "bad", "bad", "ap_level"),
    "e9c04e7e-c075-4ee3-9750-dff255ec7e84": ("bad", "good", "good", "good", "track"),
    "ecd07b7e-6450-4e31-bef1-f195129eb3d3": ("good", "bad", "good", "good", "lf_level"),
    "edc82c06-83fd-44be-a41e-52fdeea728b4": (
        "bad",
        "bad",
        "good",
        "bad",
        "lf_level;wf_shape;depth;track",
    ),
    "ef7e9f3e-a53a-4e00-8f6a-537667af2bea": (
        "bad",
        "good",
        "good",
        "good",
        "depth;track;swap",
    ),
    "f083fcd5-e456-433d-8aa5-6ace0a7ac170": ("bad", "good", "good", "good", "track"),
    "f4f67dbd-088c-4583-9477-c4cc460a8aec": ("review", "good", "good", "good", "track"),
    "f72e60c3-3593-466a-afaf-91145c44fb2b": ("good", "good", "good", "bad", "wf_shape"),
    "f93bfce4-e814-4ae3-9cdf-59f4dcdedf51": ("good", "good", "bad", "bad", "channels"),
    "fb0de39b-c750-4adc-af18-12bf8a22ed41": ("good", "bad", "good", "good", "lf_level"),
    "fb2f994b-5be8-4c1c-81dd-aad4b19ddcf7": (
        "good",
        "bad",
        "good",
        "review",
        "lf_level;lf_shape",
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
    "ff5fceaf-8247-42ab-a187-f7876441899d": ("bad", "good", "good", "good", "track"),
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
