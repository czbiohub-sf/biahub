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


# Command list: (name, import_path, help)
COMMANDS = [
    (
        "estimate_bleaching",
        "biahub.estimate_bleaching.estimate_bleaching_cli",
        "Estimate bleaching from raw data",
    ),
    (
        "estimate_deskew",
        "biahub.estimate_deskew.estimate_deskew_cli",
        "Routine for estimating deskewing parameters",
    ),
    ("deskew", "biahub.deskew.deskew_cli", "Deskew a single position across T and C axes"),
    (
        "estimate_registration",
        "biahub.estimate_registration.estimate_registration_cli",
        "Estimate affine transform between timepoints or arms",
    ),
    (
        "optimize_registration",
        "biahub.optimize_registration.optimize_registration_cli",
        "Optimize transform based on match filtering",
    ),
    (
        "register",
        "biahub.register.register_cli",
        "Apply an affine transformation to a single position",
    ),
    (
        "estimate_stitch",
        "biahub.estimate_stitch.estimate_stitch_cli",
        "Estimate stitching parameters for positions",
    ),
    ("stitch", "biahub.stitch.stitch_cli", "Stitch positions in wells of a zarr store"),
    (
        "concatenate",
        "biahub.concatenate.concatenate_cli",
        "Concatenate datasets (with optional cropping)",
    ),
    (
        "estimate_stabilization",
        "biahub.estimate_stabilization.estimate_stabilization_cli",
        "Estimate translation matrices for XYZ stabilization",
    ),
    (
        "stabilize",
        "biahub.stabilize.stabilize_cli",
        "Apply stabilization transforms to dataset",
    ),
    (
        "estimate_crop",
        "biahub.estimate_crop.estimate_crop_cli",
        "Estimate crop region for dual-channel alignment",
    ),
    (
        "compute_transfer_function",
        "biahub.compute_transfer_function.compute_transfer_function_cli",
        "Compute transfer function using PSF",
    ),
    (
        "apply_inverse_transfer_function",
        "biahub.apply_inverse_transfer_function.apply_inverse_transfer_function_cli",
        "Apply inverse transfer function to dataset",
    ),
    (
        "reconstruct",
        "biahub.reconstruct.reconstruct_cli",
        "Reconstruct a dataset using config",
    ),
    (
        "estimate_psf",
        "biahub.estimate_psf.estimate_psf_cli",
        "Estimate point spread function from beads",
    ),
    (
        "deconvolve",
        "biahub.deconvolve.deconvolve_cli",
        "Deconvolve across T and C axes using a PSF",
    ),
    (
        "characterize_psf",
        "biahub.characterize_psf.characterize_psf_cli",
        "Characterize point spread function (PSF)",
    ),
    (
        "segment",
        "biahub.segment.segment_cli",
        "Segment a position using pretrained model or pipeline",
    ),
    ("virtual_stain", "biahub.virtual_stain.virtual_stain_cli", "Run VisCy virtual staining"),
    (
        "process_with_config",
        "biahub.process_data.process_with_config_cli",
        "Process data with YAML-defined functions",
    ),
    ("track", "biahub.track.track_cli", "Track objects in 2D/3D time-lapse microscopy"),
]


# Register lazy commands
for name, import_path, help_text in COMMANDS:
    cli.add_command(
        click.Command(
            name=name,
            callback=lazy_import_command(import_path),
            help=help_text,
            short_help=help_text.split(".")[0],
        )
    )


if __name__ == "__main__":
    cli()
