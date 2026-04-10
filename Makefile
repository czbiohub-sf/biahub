PACKAGE_NAME := biahub

.PHONY: setup-develop
setup-develop:
	uv sync --group dev

.PHONY: uninstall
uninstall:
	pip uninstall -y $(PACKAGE_NAME)

.PHONY: check-format
check-format:
	ruff format --check .

.PHONY: format
format:
	ruff format .

.PHONY: lint
lint:
	ruff check .

.PHONY: lint-fix
lint-fix:
	ruff check --fix .

# run the pre-commit hooks on all files (not just staged changes)
# (requires pre-commit to be installed)
.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	uv run pytest --disable-pytest-warnings
