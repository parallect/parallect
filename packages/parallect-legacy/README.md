# parallect (compatibility shim)

> **Heads up:** The PyPI distribution `parallect` has been renamed to
> **`parallect-cli`** as of 0.3.0. This package exists only as a
> backward-compatibility shim so that `pip install parallect` keeps working.

## What this package does

This wheel does two things:

1. Depends on `parallect-cli==0.3.0`, so installing it transparently
   installs the real package. Everything still works:

   ```bash
   pip install parallect        # installs parallect-cli under the hood
   parallect research "…"       # CLI command is unchanged
   python -c "import parallect" # import name is unchanged
   ```

2. Installs a tiny sentinel module (`parallect_deprecation`) which
   `parallect-cli` detects on import and uses to emit a
   `DeprecationWarning`, so you know to migrate your install command.

## What you should do

Update your `pip install` command in `requirements.txt`,
`pyproject.toml`, CI scripts, Dockerfiles, and docs from:

```diff
-pip install parallect
+pip install parallect-cli
```

Nothing else needs to change. The import name (`import parallect`), the
CLI entry point (`parallect …`), the configuration directory
(`~/.config/parallect/`), and the public API are **all unchanged**.

## Timeline

- **0.3.0 (this release):** legacy name installs the canonical package
  and emits a `DeprecationWarning` on import.
- **Future release (3–6 months out):** legacy name will hard-error at
  install time, directing users to `pip install parallect-cli`.

The `parallect` name will never be deleted, yanked, or transferred.

## Why the rename?

The bare name `parallect` is overloaded across the ecosystem (brand,
hosted SaaS, OSS CLI, Claude Code plugin). Separating the CLI into its
own distribution name (`parallect-cli`) frees the bare `parallect` name
for the umbrella brand, while keeping `import parallect` and the
`parallect` shell command stable for existing users.

See [parallect/prx-ecosystem#3](https://github.com/parallect/prx-ecosystem/issues/3)
for context.

## Canonical package

- PyPI: https://pypi.org/project/parallect-cli/
- Source: https://github.com/parallect/parallect
