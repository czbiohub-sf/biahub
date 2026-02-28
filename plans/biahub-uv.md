# Plan: Migrate biahub packaging from setuptools to uv

## Goal

Convert biahub from setuptools to uv as the package manager. Single package — no workspaces.

## Changes

### 1. Replace build system in pyproject.toml

- Replace `[build-system]` from setuptools to hatchling (uv's default)
- Remove `[tool.setuptools]` and `[tool.setuptools.dynamic]` sections
- Set a static version or use hatch-vcs for versioning
- Keep all existing dependencies, scripts, optional-dependencies unchanged

### 2. Add uv-specific configuration

- Add `[tool.uv]` section if needed (e.g., for git dependency overrides)
- Handle git dependencies (`stitch`, `waveorder`) — may need `[tool.uv.sources]` for these

### 3. Generate lockfile

- Run `uv lock` to generate `uv.lock`
- Add `uv.lock` to version control

### 4. Verify

```bash
uv sync --all-extras
biahub --help
pytest tests/
```

## Notes

- No directory moves — `biahub/` package stays where it is
- All imports unchanged
- Click CLI entry point unchanged
- Existing `settings/` YAML configs unchanged
