# Contributing guide

Thanks for your interest in contributing to `biahub`!

## Getting started

Please read the [README](./README.md) for an overview of the project,
and how you can install and use the package.

## Making changes

Any change made to the `main` branch or release maintenance branches
need to be proposed in a [pull request](https://github.com/czbiohub/biahub/pulls) (PR).

Follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub/biahub/fork) the repository.

## Setting up a development environment

1. Install the package in development mode:

```
uv sync --group dev
```

2. Install pre-commit hooks:

```
pre-commit install
```

The pre-commit hooks automatically run style checks (via `ruff`) when staged changes are committed. Resolve any violations before committing your changes. You can manually run the pre-commit hooks at any time using the `make pre-commit` command.

## Makefile

A [makefile](Makefile) is included to help with a few basic development commands. Currently, the following commands are available:

```sh
make setup-develop # setup the package in development mode
make uninstall # uninstall the package
make check-format # run ruff format check
make format # run ruff formatting
make lint # run ruff linting
make lint-fix # run ruff linting with auto-fix
make pre-commit # run pre-commit hooks on all files
make test # run pytest
```

## Building documentation locally

The documentation is built using [Zensical](https://zensical.org). To preview the docs locally:

1. Install Zensical:

```sh
pip install zensical
```

2. Serve the documentation locally:

```sh
zensical serve
```

This will start a local server at `http://localhost:8000` with live reload enabled.

3. To build the static site:

```sh
zensical build
```

The built site will be in the `site/` directory.
