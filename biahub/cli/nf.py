import click

from iohub.ngff import open_ome_zarr


@click.group("nf")
def nf_cli():
    """Nextflow-oriented utility commands.

    Generic helpers shared across Nextflow pipelines. Step-specific init/run
    logic lives on each step's own CLI command (e.g. ``biahub deskew``).
    """


@nf_cli.command("list-positions")
@click.option("--input-zarr", "-i", required=True, type=click.Path(exists=True))
def list_positions(input_zarr: str):
    """List position keys in a plate zarr (one per line, for Nextflow fan-out)."""
    with open_ome_zarr(input_zarr, mode="r") as plate:
        for name, _ in plate.positions():
            click.echo(name)
