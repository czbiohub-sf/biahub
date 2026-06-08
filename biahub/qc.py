from pathlib import Path

import click
import yaml


def generate_report_spec(
    output: str,
    zarr_paths: tuple[str, ...],
    config_dir: str | None = None,
    title: str = "QC Report",
) -> Path:
    """Generate a report-spec YAML for imaging-qc from completed zarr stores.

    Each zarr path becomes a tab. Labels are derived from the parent directory
    name; qc_dir is the external sibling ``<stem>_qc/`` directory.
    """
    tabs = []
    for zarr_path in zarr_paths:
        p = Path(zarr_path)
        stem = p.name.removesuffix(".zarr").removesuffix(".ome")
        qc_dir = str(p.parent / f"{stem}_qc")
        label = p.parent.name
        tab: dict[str, str] = {
            "label": label,
            "zarr_path": str(p),
            "qc_dir": qc_dir,
        }
        if config_dir:
            tab["config"] = config_dir
        tabs.append(tab)

    spec = {"title": title, "tabs": tabs}
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(spec, sort_keys=False))
    return out_path


@click.command("generate-report-spec")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output YAML path.")
@click.option(
    "--config-dir",
    type=click.Path(exists=True),
    default=None,
    help="QC config directory (shared across all tabs).",
)
@click.option("--title", default="QC Report", help="Report title.")
@click.argument("zarr_paths", nargs=-1, required=True)
def generate_report_spec_cli(
    output: str, config_dir: str | None, title: str, zarr_paths: tuple[str, ...]
):
    r"""Generate a report-spec YAML for imaging-qc from completed zarr stores.

    \b
    Each zarr path becomes a tab with an auto-derived label and qc_dir:
    >>> biahub generate-report-spec -o spec.yaml /data/0-flatfield/plate.zarr /data/2-reconstruct/plate.zarr
    """
    out_path = generate_report_spec(
        output=output,
        zarr_paths=zarr_paths,
        config_dir=config_dir,
        title=title,
    )
    click.echo(str(out_path))


if __name__ == "__main__":
    generate_report_spec_cli()
