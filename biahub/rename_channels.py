from pathlib import Path

import click

from iohub.ngff import open_ome_zarr


def rename_channels(
    input_zarr: str,
    position: str,
    prefix: str = "",
    suffix: str = "",
) -> None:
    """Rename channels for a single position (metadata-only, no data copy).

    Parameters
    ----------
    input_zarr : str
        Path to the HCS plate zarr store.
    position : str
        Position key like ``B/3/000000``.
    prefix : str
        Prefix to prepend to each channel name.
    suffix : str
        Suffix to append to each channel name.
    """
    if not prefix and not suffix:
        raise click.ClickException("Provide at least --prefix or --suffix.")

    position_path = Path(input_zarr) / position
    with open_ome_zarr(str(position_path), mode="r+") as pos:
        for old_name in list(pos.channel_names):
            new_name = f"{prefix}{old_name}{suffix}"
            pos.rename_channel(old_name, new_name)

    click.echo(f"Renamed channels: {position}")


@click.command("rename-channels")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
@click.option("--position", "-p", required=True)
@click.option("--prefix", default="", help="Prefix to prepend to each channel name.")
@click.option("--suffix", default="", help="Suffix to append to each channel name.")
def rename_channels_cli(input_zarr: str, position: str, prefix: str, suffix: str):
    r"""Rename channels for a single position (metadata-only, no data copy).

    \b
    Add a prefix to all channel names:
    >>> biahub rename-channels -i ./reconstruct.zarr -p B/3/000000 --prefix "ls_"

    \b
    Add a suffix:
    >>> biahub rename-channels -i ./reconstruct.zarr -p B/3/000000 --suffix "_raw"
    """
    rename_channels(
        input_zarr=input_zarr,
        position=position,
        prefix=prefix,
        suffix=suffix,
    )


if __name__ == "__main__":
    rename_channels_cli()
