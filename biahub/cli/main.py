import importlib

import click

CONTEXT = {"help_option_names": ["-h", "--help"]}


class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return list(self.commands.keys())


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """command-line tools for biahub"""


class LazyCommand(click.Command):
    def __init__(self, name, import_path, help=None, short_help=None):
        self.import_path = import_path
        self._real_command = None
        super().__init__(name=name, help=help, short_help=short_help)

    def _load_real_command(self):
        if self._real_command is None:
            module_path, attr_name = self.import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self._real_command = getattr(module, attr_name)

    def invoke(self, ctx):
        self._load_real_command()
        return self._real_command.invoke(ctx)

    def get_help(self, ctx):
        self._load_real_command()
        return self._real_command.get_help(ctx)

    def get_params(self, ctx):
        self._load_real_command()
        return self._real_command.get_params(ctx)

    def format_usage(self, ctx, formatter):
        self._load_real_command()
        return self._real_command.format_usage(ctx, formatter)

    def format_options(self, ctx, formatter):
        self._load_real_command()
        return self._real_command.format_options(ctx, formatter)


COMMANDS = [
    {
        "name": "estimate-bleaching",
        "import_path": "biahub.estimate_bleaching.estimate_bleaching_cli",
        "help": "Estimate bleaching from raw data",
    },
    {
        "name": "estimate-deskew",
        "import_path": "biahub.estimate_deskew.estimate_deskew_cli",
        "help": "Routine for estimating deskewing parameters",
    },
    {
        "name": "deskew",
        "import_path": "biahub.deskew.deskew_cli",
        "help": "Deskew a single position across T and C axes",
    },
    {
        "name": "estimate-registration",
        "import_path": "biahub.estimate_registration.estimate_registration_cli",
        "help": "Estimate affine transform between timepoints or arms",
    },
    {
        "name": "optimize-registration",
        "import_path": "biahub.optimize_registration.optimize_registration_cli",
        "help": "Optimize transform based on match filtering",
    },
    {
        "name": "register",
        "import_path": "biahub.register.register_cli",
        "help": "Apply an affine transformation to a single position",
    },
    {
        "name": "estimate-stitch",
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
        "name": "estimate-stabilization",
        "import_path": "biahub.estimate_stabilization.estimate_stabilization_cli",
        "help": "Estimate translation matrices for XYZ stabilization",
    },
    {
        "name": "stabilize",
        "import_path": "biahub.stabilize.stabilize_cli",
        "help": "Apply stabilization transforms to dataset",
    },
    {
        "name": "estimate-crop",
        "import_path": "biahub.estimate_crop.estimate_crop_cli",
        "help": "Estimate crop region for dual-channel alignment",
    },
    {
        "name": "compute-tf",
        "import_path": "biahub.compute_transfer_function.compute_transfer_function_cli",
        "help": "Compute transfer function using PSF",
    },
    {
        "name": "apply-inv-tf",
        "import_path": "biahub.apply_inverse_transfer_function.apply_inverse_transfer_function_cli",
        "help": "Apply inverse transfer function to dataset",
    },
    {
        "name": "reconstruct",
        "import_path": "biahub.reconstruct.reconstruct_cli",
        "help": "Reconstruct a dataset using config",
    },
    {
        "name": "estimate-psf",
        "import_path": "biahub.estimate_psf.estimate_psf_cli",
        "help": "Estimate point spread function from beads",
    },
    {
        "name": "deconvolve",
        "import_path": "biahub.deconvolve.deconvolve_cli",
        "help": "Deconvolve across T and C axes using a PSF",
    },
    {
        "name": "characterize-psf",
        "import_path": "biahub.characterize_psf.characterize_psf_cli",
        "help": "Characterize point spread function (PSF)",
    },
    {
        "name": "segment",
        "import_path": "biahub.segment.segment_cli",
        "help": "Segment a position using pretrained model or pipeline",
    },
    {
        "name": "virtual-stain",
        "import_path": "biahub.virtual_stain.virtual_stain_cli",
        "help": "Run VisCy virtual staining",
    },
    {
        "name": "process-with-config",
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
        LazyCommand(
            name=cmd["name"],
            import_path=cmd["import_path"],
            help=cmd["help"],
            short_help=cmd["help"].split(".")[0],
        )
    )


if __name__ == "__main__":
    cli()
