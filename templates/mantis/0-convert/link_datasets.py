# %%
import os
from pathlib import Path

from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta

# %%
# Set dataset name in current shell with `export DATASET='dataset_name`
dataset = os.environ.get("DATASET")
if dataset is None:
    print("$DATASET environmental variable is not set")
    exit(1)

start_chunk = 1
# get script path
path = Path(os.path.realpath(__file__))


raw_data_path = Path("/hpc/instruments/cm.mantis")
projects_path = path.parent.parent.parent

print("Raw data path:", raw_data_path)
print("Projects path:", projects_path)

source_path = raw_data_path/ dataset/ "0-convert"
dest_path = projects_path / dataset / "0-convert"
print("Source path:", source_path)
print("Destination path:", dest_path)

files = os.listdir(source_path)
# filter ones with chunked in the name
files = [f for f in files if "chunked" in f]
end_chunk = len(files)
print("Number of chunk:", end_chunk)
well_map_path = raw_data_path / dataset / "well_map.csv"
# copy to the destination folder
os.system(f"cp {well_map_path} {dest_path}")

modalities = ["labelfree", "lightsheet"]

for modality in modalities:

    all_positions = []
    all_channels = []
    shapes = []
    chunks = []
    scales = []
    timepoints = []
    dtypes = []
    for chk in range(start_chunk, end_chunk + 1):
        ds_chunks = open_ome_zarr(
            source_path
            / (dataset + f"_chunked_{chk}")
            / f"{dataset}_chunked_{modality}_1.zarr",
            mode="r",
        )
        all_positions.extend([p[0] for p in ds_chunks.positions()])
        all_channels.extend(ds_chunks.channel_names)
        _, pos = next(ds_chunks.positions())
        shapes.append(pos.data.shape)
        chunks.append(pos.data.chunks)
        scales.append(pos.scale)
        timepoints.append(pos.data.shape[0])
        dtypes.append(pos.data.dtype)

    all_positions = list(dict.fromkeys(all_positions))
    all_channels = list(dict.fromkeys(all_channels))
    # assert(all([s == shapes[0] for s in shapes]))
    assert all([c == chunks[0] for c in chunks])
    assert all([s == scales[0] for s in scales])
    assert all([d == dtypes[0] for d in dtypes])

    czyx_shape = shapes[0][1:]
    chunk_size = chunks[0]
    scale = scales[0]
    dtype = dtypes[0]
    sizeT = sum(timepoints)

    # note: MicroManager metadata is not transferred
    ds = open_ome_zarr(
        dest_path / (dataset + "_symlink") / (dataset + f"_{modality}_1.zarr"),
        layout="hcs",
        mode="w-",
        channel_names=all_channels,
    )
    for position in all_positions:
        pos = ds.create_position(*position.split("/"))
        pos.create_zeros(
            name="0",
            shape=(sizeT,) + czyx_shape,
            chunks=chunk_size,
            dtype=dtype,
            transform=[TransformationMeta(type="scale", scale=scale)],
        )

    for position in all_positions:
        t = 0
        for chk, t_chk in zip(range(start_chunk, end_chunk + 1), timepoints):
            for _t in range(t_chk):
                src = Path(
                    source_path,
                    f"{dataset}_chunked_{chk}",
                    f"{dataset}_chunked_{modality}_1.zarr",
                    Path(position, "0", str(_t)),
                )
                dst = Path(
                    dest_path,
                    f"{dataset}_symlink",
                    f"{dataset}_{modality}_1.zarr",
                    Path(position, "0", str(t)),
                )
                os.symlink(src, dst, target_is_directory=True)
                t += 1

    print(f"Symlinked {modality} data for {dataset}")
print("Done")
