import click
import yaml


def echo_settings(settings):
    click.echo(yaml.dump(settings.model_dump(), default_flow_style=False, sort_keys=False))


def echo_headline(headline):
    click.echo(click.style(headline, fg="green"))
