"""
Hydra-based CLI for biahub.

This provides a new entry point using Hydra for configuration management,
enabling config composition, CLI overrides, and structured logging.

Usage examples:
    # Run with default config
    biahub-hydra

    # Override config values from command line
    biahub-hydra registration.target_channel_name=Phase3D

    # Use different config group
    biahub-hydra registration=beads

    # Run workflow
    biahub-hydra workflow=lightsheet_pipeline data.input_path=./data.zarr

    # Parameter sweep (multirun)
    biahub-hydra --multirun deskew.ls_angle_deg=28,30,32
"""
import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from biahub.hydra_configs import BiahubConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for Hydra-based biahub CLI.

    Args:
        cfg: Hydra configuration (automatically loaded and composed)

    Returns:
        Optional metric value for hyperparameter optimization
    """
    log.info("=" * 80)
    log.info("Starting biahub with Hydra configuration")
    log.info("=" * 80)

    # Log the full configuration
    log.info(f"Working directory: {Path.cwd()}")
    log.info(f"Output directory: {cfg.output_dir}")
    log.info("\nConfiguration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Validate configuration by converting to structured config
    try:
        validated_cfg = OmegaConf.to_object(cfg)
        log.info("✓ Configuration validated successfully")
    except Exception as e:
        log.error(f"✗ Configuration validation failed: {e}")
        raise

    # Execute workflow if specified
    if "workflow" in cfg and cfg.workflow is not None:
        execute_workflow(cfg)
    # Execute single command based on what's configured
    elif "registration" in cfg and cfg.registration is not None:
        execute_registration(cfg)
    elif "deskew" in cfg and cfg.deskew is not None:
        execute_deskew(cfg)
    else:
        log.warning("No workflow or command specified in configuration")
        log.info("Available commands: registration, deskew, workflow")

    log.info("=" * 80)
    log.info("biahub execution completed")
    log.info("=" * 80)

    return None


def execute_workflow(cfg: DictConfig):
    """Execute a multi-step workflow"""
    log.info(f"Executing workflow: {cfg.workflow.name}")
    log.info(f"Steps: {cfg.workflow.steps}")

    for step in cfg.workflow.steps:
        log.info(f"→ Running step: {step}")

        if step == "deskew":
            execute_deskew(cfg)
        elif step == "estimate_registration":
            execute_registration(cfg)
        elif step == "register":
            log.info("  Applying registration transforms...")
            # TODO: Implement
        elif step == "stitch":
            log.info("  Stitching positions...")
            # TODO: Implement
        else:
            log.warning(f"  Unknown step: {step}")

        log.info(f"✓ Step completed: {step}")


def execute_registration(cfg: DictConfig):
    """Execute registration estimation"""
    log.info("Executing registration estimation")
    log.info(f"  Method: {cfg.registration.estimation_method}")
    log.info(f"  Target channel: {cfg.registration.target_channel_name}")
    log.info(f"  Source channel: {cfg.registration.source_channel_name}")

    # Import the actual registration function
    from biahub.estimate_registration import estimate_registration

    # Convert config to parameters
    # TODO: Map Hydra config to function parameters
    log.info("  Registration estimation configured")
    log.info("  Note: Full implementation requires mapping config to function call")


def execute_deskew(cfg: DictConfig):
    """Execute deskew operation"""
    log.info("Executing deskew")
    log.info(f"  Pixel size: {cfg.deskew.pixel_size_um} μm")
    log.info(f"  Light sheet angle: {cfg.deskew.ls_angle_deg}°")
    log.info(f"  Average slices: {cfg.deskew.average_n_slices}")

    # Import the actual deskew function
    # from biahub.deskew import deskew_cli

    # TODO: Map Hydra config to function parameters
    log.info("  Deskew configured")
    log.info("  Note: Full implementation requires mapping config to function call")


def print_config_template(command: str):
    """Print a config template for a specific command"""
    templates = {
        "registration": """
# Registration Config Template
registration:
  target_channel_name: Phase3D
  source_channel_name: GFP
  estimation_method: manual  # Options: manual, beads, ants
  verbose: false
""",
        "deskew": """
# Deskew Config Template
deskew:
  pixel_size_um: 0.115
  ls_angle_deg: 30.0
  average_n_slices: 3
  keep_overhang: false
""",
    }

    if command in templates:
        print(templates[command])
    else:
        print(f"No template available for: {command}")


if __name__ == "__main__":
    main()
