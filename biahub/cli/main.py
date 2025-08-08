import importlib

import click

CONTEXT = {"help_option_names": ["-h", "--help"]}


class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return list(self.commands.keys())


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for biahub"""


def lazy_import_command(import_path):
    def callback(*args, **kwargs):
        module_path, attr_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        command = getattr(module, attr_name)
        return command.main(args=args, standalone_mode=True)

    return callback


COMMANDS = [
    {
        "name": "estimate_bleaching",
        "import_path": "biahub.estimate_bleaching.estimate_bleaching_cli",
        "help": "Estimate bleaching from raw data",
    },
    {
        "name": "estimate_deskew",
        "import_path": "biahub.estimate_deskew.estimate_deskew_cli",
        "help": "Routine for estimating deskewing parameters",
    },
    {
        "name": "deskew",
        "import_path": "biahub.deskew.deskew_cli",
        "help": "Deskew a single position across T and C axes",
    },
    {
        "name": "estimate_registration",
        "import_path": "biahub.estimate_registration.estimate_registration_cli",
        "help": "Estimate affine transform between timepoints or arms",
    },
    {
        "name": "optimize_registration",
        "import_path": "biahub.optimize_registration.optimize_registration_cli",
        "help": "Optimize transform based on match filtering",
    },
    {
        "name": "register",
        "import_path": "biahub.register.register_cli",
        "help": "Apply an affine transformation to a single position",
    },
    {
        "name": "estimate_stitch",
        "import_path": "biahub.estimate_stitch.estimate_stitch_cli",
        "help": "Estimate stitching parameters for positions",
    },
    {
        "name": "stitch",
        "import_path": "biahub.stitch.stitch_cli",
        "help": "Stitch positions in wells of a zarr store",
    },
    {
        "name": "concatenate",
        "import_path": "biahub.concatenate.concatenate_cli",
        "help": "Concatenate datasets (with optional cropping)",
    },
    {
        "name": "estimate_stabilization",
        "import_path": "biahub.estimate_stabilization.estimate_stabilization_cli",
        "help": "Estimate translation matrices for XYZ stabilization",
    },
    {
        "name": "stabilize",
        "import_path": "biahub.stabilize.stabilize_cli",
        "help": "Apply stabilization transforms to dataset",
    },
    {
        "name": "estimate_crop",
        "import_path": "biahub.estimate_crop.estimate_crop_cli",
        "help": "Estimate crop region for dual-channel alignment",
    },
    {
        "name": "compute_transfer_function",
        "import_path": "biahub.compute_transfer_function.compute_transfer_function_cli",
        "help": "Compute transfer function using PSF",
    },
    {
        "name": "apply_inverse_transfer_function",
        "import_path": "biahub.apply_inverse_transfer_function.apply_inverse_transfer_function_cli",
        "help": "Apply inverse transfer function to dataset",
    },
    {
        "name": "reconstruct",
        "import_path": "biahub.reconstruct.reconstruct_cli",
        "help": "Reconstruct a dataset using config",
    },
    {
        "name": "estimate_psf",
        "import_path": "biahub.estimate_psf.estimate_psf_cli",
        "help": "Estimate point spread function from beads",
    },
    {
        "name": "deconvolve",
        "import_path": "biahub.deconvolve.deconvolve_cli",
        "help": "Deconvolve across T and C axes using a PSF",
    },
    {
        "name": "characterize_psf",
        "import_path": "biahub.characterize_psf.characterize_psf_cli",
        "help": "Characterize point spread function (PSF)",
    },
    {
        "name": "segment",
        "import_path": "biahub.segment.segment_cli",
        "help": "Segment a position using pretrained model or pipeline",
    },
    {
        "name": "virtual_stain",
        "import_path": "biahub.virtual_stain.virtual_stain_cli",
        "help": "Run VisCy virtual staining",
    },
    {
        "name": "process_with_config",
        "import_path": "biahub.process_data.process_with_config_cli",
        "help": "Process data with YAML-defined functions",
    },
    {
        "name": "track",
        "import_path": "biahub.track.track_cli",
        "help": "Track objects in 2D/3D time-lapse microscopy",
    },
]


for cmd in COMMANDS:
    cli.add_command(
        click.Command(
            name=cmd["name"],
            callback=lazy_import_command(cmd["import_path"]),
            help=cmd["help"],
            short_help=cmd["help"].split(".")[0],
        )
    )


if __name__ == "__main__":
    cli()
