import pytest

from biahub.channel_names import canonical_channel_name


@pytest.mark.parametrize(
    "name, expected",
    [
        ("BF - Oblique", "BF"),
        ("BF", "BF"),
        ("Phase3D", "Phase3D"),
        ("Phase3D_bg", "Phase3D"),
        ("nuclei", "nuclei_prediction"),
        ("membrane", "membrane_prediction"),
        ("nuclei_prediction", "nuclei_prediction"),
        ("GFP EX488 EM525-45", "raw GFP EX488 EM525-45"),
        ("raw GFP EX488 EM525-45", "raw GFP EX488 EM525-45"),
    ],
)
def test_canonical_channel_name(name, expected):
    assert canonical_channel_name(name) == expected
