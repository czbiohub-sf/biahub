"""
CLI commands for managing analysis templates.

Provides commands to list, describe, and initialize projects from templates.
"""
import shutil
from pathlib import Path

import click


@click.group(name="template")
def template_cli():
    """Manage analysis pipeline templates"""
    pass


@template_cli.command(name="list")
def list_templates():
    """List available analysis templates"""
    templates_dir = Path(__file__).parent.parent.parent / "templates"

    if not templates_dir.exists():
        click.echo("No templates directory found.")
        return

    templates = [d for d in templates_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not templates:
        click.echo("No templates available.")
        return

    click.echo("Available templates:\n")
    for template in sorted(templates):
        readme = template / "README.md"
        description = "No description available"

        if readme.exists():
            # Try to extract first line or paragraph from README
            with open(readme) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    description = lines[1].strip()

        click.echo(f"  • {template.name}")
        click.echo(f"    {description}\n")


@template_cli.command(name="info")
@click.argument("template_name")
def template_info(template_name: str):
    """Show detailed information about a template"""
    templates_dir = Path(__file__).parent.parent.parent / "templates"
    template_path = templates_dir / template_name

    if not template_path.exists():
        click.echo(f"Template '{template_name}' not found.", err=True)
        click.echo(f"\nAvailable templates:")
        list_templates.callback()
        return

    readme = template_path / "README.md"
    if readme.exists():
        with open(readme) as f:
            content = f.read()
            # Show first 50 lines
            lines = content.split('\n')[:50]
            click.echo('\n'.join(lines))
            if len(content.split('\n')) > 50:
                click.echo("\n... (see README.md for full documentation)")
    else:
        click.echo(f"No documentation found for template '{template_name}'")


@template_cli.command(name="init")
@click.option(
    "-t", "--template",
    required=True,
    help="Template name to use"
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(),
    help="Output directory for new project"
)
@click.option(
    "--dataset",
    help="Dataset name (sets DATASET environment variable in scripts)"
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite output directory if it exists"
)
def init_template(template: str, output: str, dataset: str = None, force: bool = False):
    """Initialize a new project from a template

    Examples:

        # Create new project from mantis template
        biahub template init -t mantis -o ./my-experiment --dataset exp_2026

        # Use dragonfly template
        biahub template init -t dragonfly -o ./dragonfly-data --dataset dragon_001
    """
    templates_dir = Path(__file__).parent.parent.parent / "templates"
    template_path = templates_dir / template
    output_path = Path(output)

    # Validate template
    if not template_path.exists():
        click.echo(f"Error: Template '{template}' not found.", err=True)
        click.echo(f"\nAvailable templates:")
        list_templates.callback()
        return 1

    # Check output directory
    if output_path.exists() and not force:
        click.echo(f"Error: Output directory '{output}' already exists.", err=True)
        click.echo("Use --force to overwrite.")
        return 1

    # Copy template
    click.echo(f"Initializing project from '{template}' template...")
    click.echo(f"Output directory: {output_path.absolute()}")

    try:
        if output_path.exists():
            shutil.rmtree(output_path)

        shutil.copytree(template_path, output_path, ignore=shutil.ignore_patterns('.git*'))

        click.echo(f"✓ Template copied successfully!")

        # If dataset name provided, create a setup script
        if dataset:
            setup_script = output_path / "setup_dataset.sh"
            with open(setup_script, 'w') as f:
                f.write(f"#!/bin/bash\n")
                f.write(f"# Auto-generated dataset setup script\n\n")
                f.write(f"export DATASET={dataset}\n")
                f.write(f"echo \"Dataset set to: $DATASET\"\n")
            setup_script.chmod(0o755)
            click.echo(f"✓ Created setup script: {setup_script}")
            click.echo(f"  Run: source {setup_script}")

        # Show next steps
        click.echo(f"\nNext steps:")
        click.echo(f"  1. cd {output_path}")
        if dataset:
            click.echo(f"  2. source setup_dataset.sh")
        click.echo(f"  3. Review and edit configuration files")
        click.echo(f"  4. Follow the template README for usage instructions")
        click.echo(f"\nFor help: cat {output_path}/README.md")

    except Exception as e:
        click.echo(f"Error initializing template: {e}", err=True)
        return 1


if __name__ == "__main__":
    template_cli()
