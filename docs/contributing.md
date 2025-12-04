# Contributing Guide

Thanks for your interest in contributing to `biahub`!

## Getting started

Please read the [Home](index.md) page for an overview of the project,
and how you can install and use the package.

## Making changes

Any change made to the `main` branch or release maintenance branches
need to be proposed in a [pull request](https://github.com/czbiohub-sf/biahub/pulls) (PR).

Follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub-sf/biahub/fork) the repository.

## Setting up a development environment

1. Install the package in development mode:

```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:

```bash
pre-commit install
```

The pre-commit hooks automatically run style checks (e.g. `flake8`, `black`, `isort`) when staged changes are committed. Resolve any violations before committing your changes. You can manually run the pre-commit hooks at any time using the `make pre-commit` command.

## Makefile

A [makefile](https://github.com/czbiohub-sf/biahub/blob/main/Makefile) is included to help with a few basic development commands. Currently, the following commands are available:

```bash
make setup-develop # setup the package in development mode
make uninstall # uninstall the package
make check-format # run black and isort format check
make format # run black and isort formatting
make lint # run flake8 linting
make pre-commit # run pre-commit hooks on all files
make test # run pytest
```

## Building documentation locally

The documentation is built using [Zensical](https://zensical.org). To preview the docs locally:

1. Install Zensical:

```bash
pip install zensical
```

2. Serve the documentation locally:

```bash
zensical serve
```

This will start a local server at `http://localhost:8000` with live reload enabled.

3. To build the static site:

```bash
zensical build
```

The built site will be in the `site/` directory.
