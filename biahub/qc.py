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
    name — the pipeline step the store came out of — falling back to the store's
    own stem when there is no parent name to read (a bare relative path). An
    empty label is rejected downstream by ``imaging-qc report``, and it would be
    rejected only after every compute task had already run.

    ``qc_dir`` names where that store's tables actually are. Two locations are
    possible: the store's own ``tables/qc/`` group, which is what a plain run
    writes and therefore what nextflow/qc-standalone.nf produces, and the
    external sibling ``<stem>_qc/``, which only appears when imaging-qc ran with
    ``--output-dir``. A per-tab ``qc_dir`` is required, and it *overrides*
    imaging-qc's own search of both locations — so naming the wrong one renders
    an empty tab at exit 0 rather than failing. The in-store group therefore wins
    when it exists, which is imaging-qc's own precedence in ``_stage_table_dir``.
    """
    tabs = []
    for zarr_path in zarr_paths:
        p = Path(zarr_path)
        stem = p.name.removesuffix(".zarr").removesuffix(".ome")
        in_store = p / "tables" / "qc"
        qc_dir = str(in_store if in_store.is_dir() else p.parent / f"{stem}_qc")
        label = p.parent.name or stem
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
