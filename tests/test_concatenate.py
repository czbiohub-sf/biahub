import numpy as np
import pytest

from iohub import open_ome_zarr
from pydantic import ValidationError

from biahub.concatenate import _channel_combiner_metadata, concatenate
from biahub.settings import ConcatenateSettings, TimeRange
from biahub.utils.config import model_to_yaml

# Single position for tests that don't need multiple positions
_ONE_POS = [("A", "1", "0")]


def test_concatenate_channels(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating channels across zarr stores with the same layout
    """
    # Create example plates with same layout and different channels
    position_list = ["A/1/0", "B/1/0"]
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1",
        position_list=[p.split("/") for p in position_list],
        channel_names=["DAPI", "Cy3"],
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2",
        position_list=[p.split("/") for p in position_list],
        channel_names=["GFP", "RFP", "Phase3D"],
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices="all",
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has all the channels from the input plates
    # channel ordering might be different
    output_channels = output_plate.channel_names
    assert set(output_channels) == set(plate_1.channel_names + plate_2.channel_names)

    # Check that the output plate has the right number of positions
    output_positions = [pos_name for pos_name, _ in output_plate.positions()]
    assert set(output_positions) == set(position_list)


def test_concatenate_specific_channels(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating specific channels from zarr stores
    """

    # Create test plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["DAPI", "Cy5"]
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2", position_list=_ONE_POS, channel_names=["GFP", "RFP"]
    )

    # Select only specific channels from each plate
    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=[
            ["DAPI"],
            ["GFP"],
        ],  # Only select DAPI from plate_1 and GFP from plate_2
        time_indices="all",
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has only the selected channels
    output_channels = output_plate.channel_names
    assert set(output_channels) == {"DAPI", "GFP"}


def test_concatenate_with_time_indices(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating with specific time indices
    """

    # Create test plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, time_points=5
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2", position_list=_ONE_POS, channel_names=["DAPI"], time_points=5
    )

    # Select only specific time indices
    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices=[2, 3],  # Select specific time points
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has the two time points
    assert output_plate["A/1/0"].data.shape[0] == 2


def test_concatenate_refuses_unequal_time_points(create_custom_plate, tmp_path, sbatch_file):
    """
    time_indices "all" with sources of different lengths is a crop and is refused
    """
    plate_1_path, _ = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["DAPI"], time_points=5
    )
    plate_2_path, _ = create_custom_plate(
        tmp_path / "zarr2", position_list=_ONE_POS, channel_names=["GFP"], time_points=4
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        time_indices="all",
    )
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    with pytest.raises(ValueError, match=r"time_indices: \{start: 0, stop: 4\}"):
        concatenate(
            input_position_dirpaths=None,
            config_filepath=config_path,
            output_dirpath=tmp_path / "output.zarr",
            sbatch_filepath=sbatch_file,
            cluster="debug",
            monitor=False,
        )


@pytest.mark.parametrize(
    "mode, taken, log",
    [
        ("min", [0, 1, 2, 3], "cropping"),
        ("max", [0, 1, 2, 3, 4], "padding"),
        ({"start": 0, "stop": 4}, [0, 1, 2, 3], "cropping"),
        ({"start": 0, "stop": 5}, [0, 1, 2, 3, 4], "padding"),
        ({"start": 0, "stop": 5, "step": 2}, [0, 2, 4], "padding"),
        ([1, 4], [1, 4], "padding"),
        (4, [4], "padding"),
    ],
)
def test_concatenate_uneven_time_points(
    create_custom_plate, tmp_path, sbatch_file, capsys, mode, taken, log
):
    """
    With uneven sources, every mode but "all" takes its time points from each
    source that has them and leaves the rest at zero, and logs a crop or a pad
    """
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["DAPI"], time_points=5
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2", position_list=_ONE_POS, channel_names=["GFP"], time_points=4
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        time_indices=mode,
    )
    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output = open_ome_zarr(output_path)["A/1/0"].data[:]
    assert output.shape[0] == len(taken)
    np.testing.assert_array_equal(output[:, 0], plate_1["A/1/0"].data[taken, 0])
    for i, t in enumerate(taken):
        if t < 4:
            np.testing.assert_array_equal(output[i, 1], plate_2["A/1/0"].data[t, 0])
        else:
            assert not output[i, 1].any()
    assert f"{log}" in capsys.readouterr().out


def _plate_with_time_points(path, channel_names, time_points_per_position):
    """An HCS plate whose positions have different numbers of time points."""
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=channel_names) as plate:
        for name, num_t in time_points_per_position.items():
            plate.create_position(*name.split("/"))["0"] = np.random.randint(
                1, 1000, size=(num_t, len(channel_names), 2, 3, 3), dtype=np.uint16
            )
    return open_ome_zarr(path)


@pytest.mark.parametrize(
    "mode, expected_T", [("min", 3), ("max", 5), ({"start": 0, "stop": 5}, 5)]
)
def test_concatenate_uneven_per_position_workers(tmp_path, sbatch_file, mode, expected_T):
    """
    With uneven positions, --init sizes T over every position and each
    per-position worker follows the plate's T rather than its own sources':
    under "min" position A has 5 time points but the plate keeps 3; under
    "max" and the range every source of position B is shorter than the plate's
    5, and its worker pads instead of refusing time point 4
    """
    t_by_source = [{"A/1/0": 5, "B/1/0": 4}, {"A/1/0": 5, "B/1/0": 3}]
    plates = [
        _plate_with_time_points(tmp_path / f"s{i}.zarr", [name], t_by_position)
        for i, (name, t_by_position) in enumerate(
            zip(["DAPI", "GFP"], t_by_source, strict=True)
        )
    ]
    sources = [tmp_path / "s0.zarr", tmp_path / "s1.zarr"]

    config_path = tmp_path / "concat.yml"
    model_to_yaml(ConcatenateSettings(time_indices=mode), config_path)
    output_path = tmp_path / "output.zarr"
    common = dict(
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )
    concatenate(
        input_position_dirpaths=[[src / "A/1/0", src / "B/1/0"] for src in sources],
        init_only=True,
        **common,
    )
    # One worker per position, given that position from every source
    for position in ["A/1/0", "B/1/0"]:
        concatenate(input_position_dirpaths=[[src / position] for src in sources], **common)

    output_plate = open_ome_zarr(output_path)
    for position in ["A/1/0", "B/1/0"]:
        output = output_plate[position].data[:]
        assert output.shape[0] == expected_T
        for channel, plate in enumerate(plates):
            num_t = min(plate[position].data.shape[0], expected_T)
            np.testing.assert_array_equal(
                output[:num_t, channel], plate[position].data[:num_t, 0]
            )
            assert not output[num_t:, channel].any()


def test_concatenate_with_time_range(create_custom_plate, tmp_path, sbatch_file):
    """
    A time range crops sources of different lengths to the time points asked for
    """
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["DAPI"], time_points=5
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2", position_list=_ONE_POS, channel_names=["GFP"], time_points=4
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        time_indices={"start": 1, "stop": 4},
    )
    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output = open_ome_zarr(output_path)["A/1/0"].data[:]
    assert output.shape[0] == 3
    np.testing.assert_array_equal(output[:, 0], plate_1["A/1/0"].data[1:4, 0])
    np.testing.assert_array_equal(output[:, 1], plate_2["A/1/0"].data[1:4, 0])


def test_concatenate_with_time_range_step(create_custom_plate, tmp_path, sbatch_file):
    """
    A time range with a step takes every step-th time point
    """
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["DAPI"], time_points=5
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*"],
        time_indices={"start": 0, "stop": 5, "step": 2},
    )
    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output = open_ome_zarr(output_path)["A/1/0"].data[:]
    np.testing.assert_array_equal(output, plate_1["A/1/0"].data[[0, 2, 4]])


def test_concatenate_refuses_time_range_past_end(create_custom_plate, tmp_path, sbatch_file):
    """
    A time range past the end of the longest source is refused
    """
    plate_1_path, _ = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["DAPI"], time_points=3
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*"],
        time_indices={"start": 0, "stop": 5},
    )
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    with pytest.raises(ValueError, match=r"\[3, 4\] are out of range: the longest"):
        concatenate(
            input_position_dirpaths=None,
            config_filepath=config_path,
            output_dirpath=tmp_path / "output.zarr",
            sbatch_filepath=sbatch_file,
            cluster="debug",
            monitor=False,
        )


def test_time_range_validation():
    """
    A time range needs stop > start; it parses from a mapping
    """
    assert ConcatenateSettings(time_indices={"stop": 4}).time_indices == TimeRange(
        start=0, stop=4, step=1
    )
    with pytest.raises(ValidationError):
        TimeRange(start=4, stop=4)
    with pytest.raises(ValidationError):
        TimeRange(stop=4, step=0)


def test_concatenate_with_single_slice_to_all(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating with a single slice applied to all datasets
    """
    # Create test plates with same shape
    (T, Z, Y, X) = (3, 4, 6, 8)
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1",
        position_list=_ONE_POS,
        time_points=T,
        z_size=Z,
        y_size=Y,
        x_size=X,
        channel_names=["GFP", "RFP", "Phase3D"],
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2",
        position_list=_ONE_POS,
        time_points=T,
        z_size=Z,
        y_size=Y,
        x_size=X,
        channel_names=["DAPI"],
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices="all",
        Z_slice=[0, 2],
        Y_slice=[0, 3],
        X_slice=[0, 4],
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    assert output_plate["A/1/0"].data.shape == (T, 4, 2, 3, 4)


def test_concatenate_with_cropping(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating with cropping
    """
    Z, Y, X = 4, 6, 8
    # Create example plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1",
        position_list=_ONE_POS,
        channel_names=["DAPI", "Cy5"],
        z_size=Z,
        y_size=Y,
        x_size=X,
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2",
        position_list=_ONE_POS,
        channel_names=["GFP", "RFP"],
        z_size=Z,
        y_size=Y,
        x_size=X,
    )

    # Define crop parameters
    z_start, z_end = 0, Z // 2
    y_start, y_end = 0, Y // 2
    x_start, x_end = 0, X // 2

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices="all",
        Z_slice=[[z_start, z_end], [z_start, z_end]],
        Y_slice=[[y_start, y_end], [y_start, y_end]],
        X_slice=[[x_start, x_end], [x_start, x_end]],
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has the expected cropped dimensions
    _, _, output_Z, output_Y, output_X = output_plate["A/1/0"].data.shape
    assert output_Z == z_end - z_start
    assert output_Y == y_end - y_start
    assert output_X == x_end - x_start


@pytest.mark.parametrize(
    ["version", "shards_ratio_time"],
    [["0.4", 1], ["0.5", None], ["0.5", 1], ["0.5", 2], ["0.5", 5]],
)
def test_concatenate_with_custom_chunks(
    create_custom_plate, tmp_path, sbatch_file, version, shards_ratio_time
):
    """
    Test concatenating with custom chunk sizes
    """
    # Create example plates
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1",
        position_list=_ONE_POS,
        channel_names=["DAPI", "Cy5"],
        time_points=3,
        z_size=4,
        y_size=8,
        x_size=6,
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2",
        position_list=_ONE_POS,
        channel_names=["GFP", "RFP"],
        time_points=3,
        z_size=4,
        y_size=8,
        x_size=6,
    )

    # Define custom chunk sizes
    chunks = [1, 1, 2, 4, 3]  # [C, Z, Y, X]
    if version == "0.5":
        if shards_ratio_time is None:
            shards_ratio = None
        else:
            shards_ratio = [shards_ratio_time, 1, 1, 2, 2]
    elif version == "0.4":
        shards_ratio = None

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices="all",
        chunks_czyx=chunks[1:],
        shards_ratio=shards_ratio,
        output_ome_zarr_version=version,
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)
    for pos_name, pos in output_plate.positions():
        assert pos.data.chunks == tuple(chunks)
        if version == "0.5" and shards_ratio is not None:
            assert pos.data.shards == tuple(
                c * s for c, s in zip(chunks, shards_ratio, strict=True)
            )
        np.testing.assert_array_equal(
            pos.data.numpy(),
            np.concatenate(
                [plate_1[pos_name].data.numpy(), plate_2[pos_name].data.numpy()], axis=1
            ),
        )

    # Check that the output plate has all the channels from the input plates
    output_channels = output_plate.channel_names
    assert set(output_channels) == set(plate_1.channel_names + plate_2.channel_names)


def test_concatenate_multiple_plates(create_custom_plate, tmp_path, sbatch_file):
    """
    Test merging positions from multiple plates: sources that share a channel
    name share its output channel, and each position keeps its own data
    """
    common_params = {"time_points": 3, "z_size": 4, "y_size": 5, "x_size": 6}

    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1",
        channel_names=["GFP", "RFP", "DAPI", "Cy5", "Phase3D"],
        **common_params,
    )

    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2", channel_names=["GFP", "RFP"], **common_params
    )

    plate_3_path, plate_3 = create_custom_plate(
        tmp_path / "zarr3", channel_names=["Phase3D"], **common_params
    )

    settings = ConcatenateSettings(
        concat_data_paths=[
            str(plate_1_path) + "/A/1/0",
            str(plate_2_path) + "/B/2/0",
            str(plate_3_path) + "/B/1/0",
        ],
        channel_names=["all", ["GFP", "RFP"], ["Phase3D"]],
        time_indices="all",
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    # Check that the output plate has one position per source
    output_positions = [pos_name for pos_name, _ in output_plate.positions()]
    assert sorted(output_positions) == ["A/1/0", "B/1/0", "B/2/0"]

    # Check that the output plate has the right channels
    output_channels = output_plate.channel_names
    assert output_channels == ["GFP", "RFP", "DAPI", "Cy5", "Phase3D"]

    # Check that the output plate has the right shape
    assert output_plate["A/1/0"].data.shape[0] == 3  # time points
    assert output_plate["A/1/0"].data.shape[1] == 5  # channels

    # Check that each source's channels landed in their shared output channels
    np.testing.assert_array_equal(output_plate["A/1/0"].data[:], plate_1["A/1/0"].data[:])
    np.testing.assert_array_equal(output_plate["B/2/0"].data[:, :2], plate_2["B/2/0"].data[:])
    np.testing.assert_array_equal(output_plate["B/1/0"].data[:, 4:], plate_3["B/1/0"].data[:])


def test_concatenate_refuses_overwrite(create_custom_plate, tmp_path, sbatch_file):
    """
    Two sources that would write the same channel of the same position are refused
    """
    plate_1_path, _ = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["GFP", "RFP", "DAPI"]
    )
    plate_2_path, _ = create_custom_plate(
        tmp_path / "zarr2", position_list=_ONE_POS, channel_names=["GFP", "RFP"]
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/A/1/0", str(plate_2_path) + "/A/1/0"],
        channel_names=["all", ["GFP"]],
    )
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    with pytest.raises(ValueError, match="'GFP' of output position A/1/0"):
        concatenate(
            input_position_dirpaths=None,
            config_filepath=config_path,
            output_dirpath=tmp_path / "output.zarr",
            sbatch_filepath=sbatch_file,
            cluster="debug",
            monitor=False,
        )


def test_channel_index_after_shared_channel(create_custom_plate, tmp_path):
    """
    A new channel that follows a shared one gets its own output channel
    """
    plate_1_path, _ = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, channel_names=["A", "B", "C"]
    )
    plate_2_path, _ = create_custom_plate(
        tmp_path / "zarr2", position_list=[("B", "1", "0")], channel_names=["B", "D"]
    )

    _, channel_names, input_idx, output_idx, _ = _channel_combiner_metadata(
        [[plate_1_path / "A/1/0"], [plate_2_path / "B/1/0"]], "all", ["all"] * 3
    )

    assert channel_names == ["A", "B", "C", "D"]
    assert input_idx == [[0, 1, 2], [0, 1]]
    assert output_idx == [[0, 1, 2], [1, 3]]


def test_concatenate_mismatched_with_cropping(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating zarr stores of mismatched shapes with cropping to the
    same output shape
    """
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1", position_list=_ONE_POS, time_points=3, z_size=2, y_size=3, x_size=3
    )

    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2",
        position_list=_ONE_POS,
        channel_names=["DAPI", "Cy5", "BF"],
        time_points=3,
        z_size=4,
        y_size=6,
        x_size=6,
    )

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_1_path) + "/*/*/*", str(plate_2_path) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices="all",
        Z_slice=["all", [0, 2]],
        Y_slice=["all", [0, 3]],
        X_slice=["all", [0, 3]],
    )

    output_path = tmp_path / "output.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    assert output_plate["A/1/0"].data.shape == (3, 6, 2, 3, 3)


def test_concatenate_with_mixed_slice_formats(create_custom_plate, tmp_path, sbatch_file):
    """
    Test concatenating with mixed slice formats like [[0,1], 'all']
    """
    # Create a plate with larger dimensions to test mixed slice formats
    plate_path_1, plate_1 = create_custom_plate(
        tmp_path / "large_plate_1",
        position_list=_ONE_POS,
        time_points=2,
        z_size=10,
        y_size=20,
        x_size=20,
    )
    plate_path_2, plate_2 = create_custom_plate(
        tmp_path / "large_plate_2",
        position_list=_ONE_POS,
        channel_names=["DAPI", "Cy5", "BF"],
        time_points=2,
        z_size=5,
        y_size=4,
        x_size=8,
    )

    # Define mixed slice formats
    z_slices = [[0, 5], "all"]
    y_slices = [[2, 6], "all"]
    x_slices = [[4, 12], "all"]

    settings = ConcatenateSettings(
        concat_data_paths=[str(plate_path_1) + "/*/*/*", str(plate_path_2) + "/*/*/*"],
        channel_names=["all", "all"],
        time_indices="all",
        Z_slice=z_slices,
        Y_slice=y_slices,
        X_slice=x_slices,
    )

    output_path = tmp_path / "output_mixed_slice.zarr"
    config_path = tmp_path / "concat.yml"
    model_to_yaml(settings, config_path)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path,
        output_dirpath=output_path,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate = open_ome_zarr(output_path)

    # Expect the shape to be the same
    assert output_plate["A/1/0"].data.shape[-3:] == (
        z_slices[0][1] - z_slices[0][0],
        y_slices[0][1] - y_slices[0][0],
        x_slices[0][1] - x_slices[0][0],
    )


def test_concatenate_with_unique_positions(create_custom_plate, tmp_path, sbatch_file):
    """
    Similar to test_concatenate_channels, but with ensure_unique_positions=True
    to prevent overwriting when multiple inputs have the same position names
    """
    # Create example plates with same layout and different channels
    position_list = ["A/1/0", "B/1/0"]
    plate_1_path, plate_1 = create_custom_plate(
        tmp_path / "zarr1",
        position_list=[p.split("/") for p in position_list],
        channel_names=["DAPI", "Cy5"],
    )
    plate_2_path, plate_2 = create_custom_plate(
        tmp_path / "zarr2",
        position_list=[p.split("/") for p in position_list],
        channel_names=["GFP", "RFP"],
    )

    # Now test with ensure_unique_positions=True
    settings_unique = ConcatenateSettings(
        concat_data_paths=[
            str(plate_1_path) + "/A/1/0",
            str(plate_2_path) + "/A/1/0",  # Same position name
        ],
        channel_names=["all", "all"],
        time_indices="all",
        ensure_unique_positions=True,  # Enable unique positions
    )

    output_path_unique = tmp_path / "output_unique.zarr"
    config_path_unique = tmp_path / "concat_unique.yml"
    model_to_yaml(settings_unique, config_path_unique)
    concatenate(
        input_position_dirpaths=None,
        config_filepath=config_path_unique,
        output_dirpath=output_path_unique,
        sbatch_filepath=sbatch_file,
        cluster="debug",
        monitor=False,
    )

    output_plate_unique = open_ome_zarr(output_path_unique)

    # Check that there are two positions (both inputs were preserved with unique names)
    output_positions_unique = [pos_name for pos_name, _ in output_plate_unique.positions()]
    assert len(output_positions_unique) == 2

    # The first position should keep its original name, the second should have a suffix
    assert "A/1/0" in output_positions_unique
    assert "A/1d1/0" in output_positions_unique  # Second position has a suffix

    # Check that both positions have the expected channels
    for _pos_name, pos in output_plate_unique.positions():
        # Both positions should have all channels
        assert set(pos.channel_names) == {"DAPI", "Cy5", "GFP", "RFP"}
