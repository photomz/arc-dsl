.PHONY: stubs

# make stubs: stubgen -m dsl --include-docstrings
# if no stubgen, uv pip install mypy --system
stubs:
	@if ! command -v stubgen &> /dev/null; then \
		echo "stubgen could not be found, installing mypy with uv..."; \
		uv pip install mypy --system; \
	fi
	stubgen -m dsl --include-docstrings
	stubgen -m arc_types --include-docstrings