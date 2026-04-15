#!/usr/bin/env python3
"""Wrapper for llama-model download with compatibility patching.

The upstream llama-model CLI currently imports `llama_models.cli.model.safety_models`,
while the installed package ships `llama_models.cli.safety_models`.
This wrapper installs a module alias and delegates to the official CLI entrypoint.
"""

from __future__ import annotations

import sys
import types


def main() -> int:
    try:
        import llama_models.cli.safety_models as safety_models
    except Exception as exc:
        print(f"Failed to import llama_models CLI modules: {exc}", file=sys.stderr)
        return 1

    compat_pkg = types.ModuleType("llama_models.cli.model")
    compat_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("llama_models.cli.model", compat_pkg)
    sys.modules["llama_models.cli.model.safety_models"] = safety_models

    from llama_models.cli.llama import main as llama_main

    sys.argv = ["llama-model", "download", *sys.argv[1:]]
    try:
        llama_main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
