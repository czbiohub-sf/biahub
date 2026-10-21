import pytest

from biahub.channel_names import canonical_channel_name, canonical_channel_names


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


@pytest.mark.parametrize(
    "names, final",
    [(["BF", "BF - Oblique"], "BF"), (["GFP", "raw GFP"], "raw GFP")],
)
def test_canonical_channel_names_collision(names, final):
    with pytest.raises(ValueError) as excinfo:
        canonical_channel_names(names)
    for name in names:
        assert repr(name) in str(excinfo.value)
    assert repr(final) in str(excinfo.value)


def test_canonical_channel_names_preserves_order():
    assert canonical_channel_names(["GFP", "nuclei", "BF - Oblique"]) == [
        "raw GFP",
        "nuclei_prediction",
        "BF",
    ]
