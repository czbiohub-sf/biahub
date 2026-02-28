"""
Example script demonstrating programmatic use of Hydra with biahub.

This shows how to use Hydra's Compose API to create and manipulate
configurations programmatically, which is useful for:
- Jupyter notebooks
- Python scripts
- Integration with other tools
"""

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path


def example_basic_config():
    """Load and print a basic config"""
    print("=" * 80)
    print("Example 1: Load Basic Config")
    print("=" * 80)

    # Initialize Hydra with config directory
    config_dir = Path(__file__).parent.parent / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        # Load default config
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))


def example_override_config():
    """Override config values programmatically"""
    print("\n" + "=" * 80)
    print("Example 2: Override Config Values")
    print("=" * 80)

    config_dir = Path(__file__).parent.parent / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        # Load config with overrides
        cfg = compose(
            config_name="config",
            overrides=[
                "registration=beads",
                "registration.verbose=true",
                "registration.target_channel_name=Phase3D",
            ],
        )
        print(OmegaConf.to_yaml(cfg))


def example_workflow_config():
    """Load a workflow configuration"""
    print("\n" + "=" * 80)
    print("Example 3: Load Workflow Config")
    print("=" * 80)

    config_dir = Path(__file__).parent.parent / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="config",
            overrides=[
                "workflow=lightsheet_pipeline",
                "data.input_path=./my_data.zarr",
                "data.output_path=./processed.zarr",
            ],
        )
        print(OmegaConf.to_yaml(cfg))


def example_parameter_sweep():
    """Simulate a parameter sweep"""
    print("\n" + "=" * 80)
    print("Example 4: Parameter Sweep (Simulation)")
    print("=" * 80)

    config_dir = Path(__file__).parent.parent / "conf"

    # Sweep over angles
    angles = [28, 30, 32]

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        for angle in angles:
            cfg = compose(
                config_name="config",
                overrides=[
                    "deskew=default",
                    f"deskew.ls_angle_deg={angle}",
                ],
            )
            print(f"\n--- Run with angle={angle}° ---")
            print(f"Angle: {cfg.deskew.ls_angle_deg}")
            print(f"Pixel size: {cfg.deskew.pixel_size_um}")


def example_config_modification():
    """Modify config after loading"""
    print("\n" + "=" * 80)
    print("Example 5: Modify Config After Loading")
    print("=" * 80)

    config_dir = Path(__file__).parent.parent / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="config", overrides=["registration=manual"])

        # Modify config using OmegaConf
        print("Original target channel:", cfg.registration.target_channel_name)

        # Update value
        OmegaConf.update(cfg, "registration.target_channel_name", "MyChannel")
        print("Updated target channel:", cfg.registration.target_channel_name)

        # Add new value
        OmegaConf.update(cfg, "registration.new_param", "new_value", merge=True)
        print("Added new param:", cfg.registration.new_param)


def example_merge_configs():
    """Merge multiple configs"""
    print("\n" + "=" * 80)
    print("Example 6: Merge Multiple Configs")
    print("=" * 80)

    config_dir = Path(__file__).parent.parent / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        # Load base config
        cfg1 = compose(config_name="config", overrides=["registration=manual"])

        # Load another config
        cfg2 = compose(config_name="config", overrides=["deskew=default"])

        # Merge them
        merged = OmegaConf.merge(cfg1, cfg2)

        print("Merged config has both registration and deskew:")
        print(f"Registration method: {merged.registration.estimation_method}")
        print(f"Deskew angle: {merged.deskew.ls_angle_deg}")


if __name__ == "__main__":
    example_basic_config()
    example_override_config()
    example_workflow_config()
    example_parameter_sweep()
    example_config_modification()
    example_merge_configs()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
