# Contributing to Parallect

Thanks for your interest in contributing to Parallect!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/parallect.git`
3. Install dependencies: `uv sync`
4. Run tests: `uv run pytest tests/`

## Development

- Use `uv` for dependency management
- Format code with `ruff format`
- Lint with `ruff check`
- All tests must pass before submitting a PR

## Adding a Provider

See [docs/PROVIDERS.md](docs/PROVIDERS.md) for the custom provider guide. Providers implement the `AsyncResearchProvider` protocol.

## Adding a Plugin

See [docs/PLUGINS.md](docs/PLUGINS.md) for the plugin system documentation.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include your Python version, OS, and a minimal reproduction case

## Security

If you discover a security vulnerability, please email security@parallect.ai instead of opening a public issue.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
