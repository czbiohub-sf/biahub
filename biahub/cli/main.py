import importlib

import typer

from typer.core import TyperCommand, TyperGroup
from typer.main import get_command

from biahub.cli.parsing import install_eat_all_options

CONTEXT = {"help_option_names": ["-h", "--help"]}


class LazyCommand(TyperCommand):
    """Typer command that imports its callback only when used."""

    def __init__(self, name, import_path, help=None, short_help=None):
        self.import_path = import_path
        self._real_command = None
        self._placeholder_params = []
        self._initializing = True
        super().__init__(name=name, help=help, short_help=short_help)
        self._initializing = False

    def _load_real_command(self):
        if self._real_command is None:
            module_path, attr_name = self.import_path.rsplit(".", 1)
            callback = getattr(importlib.import_module(module_path), attr_name)
            command_app = typer.Typer(add_completion=False)
            command_app.command(name=self.name)(callback)
            command = get_command(command_app)
            install_eat_all_options(command)
            self._real_command = command
        return self._real_command

    @property
    def params(self):
        if self._initializing:
            return self._placeholder_params
        return self._load_real_command().params

    @params.setter
    def params(self, value):
        self._placeholder_params = value

    def invoke(self, ctx):
        return self._load_real_command().invoke(ctx)

    def get_help(self, ctx):
        return self._load_real_command().get_help(ctx)

    def get_params(self, ctx):
        return self._load_real_command().get_params(ctx)

    def format_usage(self, ctx, formatter):
        return self._load_real_command().format_usage(ctx, formatter)

    def format_options(self, ctx, formatter):
        return self._load_real_command().format_options(ctx, formatter)


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
        "name": "flat-field",
        "import_path": "biahub.flat_field.flat_field_cli",
        "help": "Apply flat field correction to selected channels",
    },
    {
        "name": "flip",
        "import_path": "biahub.flip.flip_cli",
        "help": "Flip images in a dataset",
    },
    {
        "name": "optimize-registration",
        "import_path": "biahub.optimize_registration.optimize_registration_cli",
        "help": "Optimize transform based on match filtering",
    },
    {
        "name": "pyramid",
        "import_path": "biahub.pyramid.pyramid_cli",
        "help": "Create pyramid levels for a dataset",
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


class LazyGroup(TyperGroup):
    """Typer group that imports its application only when used."""

    def __init__(self, name, import_path, **kwargs):
        self.import_path = import_path
        self._real_group = None
        self._placeholder_commands = {}
        self._initializing = True
        super().__init__(name=name, **kwargs)
        self._initializing = False

    def _load(self):
        if self._real_group is None:
            module_path, attr_name = self.import_path.rsplit(".", 1)
            app = getattr(importlib.import_module(module_path), attr_name)
            group = get_command(app)
            if not isinstance(group, TyperGroup):
                raise TypeError(f"{self.import_path} did not produce a Typer group")
            self._real_group = group
        return self._real_group

    @property
    def commands(self):
        if self._initializing:
            return self._placeholder_commands
        return self._load().commands

    @commands.setter
    def commands(self, value):
        self._placeholder_commands = value

    def list_commands(self, ctx):
        return self._load().list_commands(ctx)

    def get_command(self, ctx, cmd_name):
        return self._load().get_command(ctx, cmd_name)

    def invoke(self, ctx):
        return self._load().invoke(ctx)

    def get_help(self, ctx):
        return self._load().get_help(ctx)


class RootGroup(TyperGroup):
    """Root group that exposes lazy commands in their declared order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for command in COMMANDS:
            self.add_command(
                LazyCommand(
                    name=command["name"],
                    import_path=command["import_path"],
                    help=command["help"],
                    short_help=command["help"].split(".")[0],
                )
            )
        self.add_command(
            LazyGroup(
                name="nf",
                import_path="biahub.cli.nf.cli",
                help="Nextflow utilities",
            )
        )

    def list_commands(self, ctx):
        return list(self.commands)


app = typer.Typer(
    name="biahub",
    cls=RootGroup,
    context_settings=CONTEXT,
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main():
    """command-line tools for biahub."""


if __name__ == "__main__":
    app()
