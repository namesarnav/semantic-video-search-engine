"""Packaging invariants that only bite outside the dev checkout.

`uv run python -m sv_engine.cli ...` works regardless of what
`[project.scripts]` says, so a broken console script stays invisible until
something invokes `sv-engine` directly -- which is exactly what the container
and the docs do.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _scripts() -> dict[str, str]:
    payload = tomllib.loads((REPO / "pyproject.toml").read_text())
    return payload["project"].get("scripts", {})


def test_there_is_a_console_script():
    assert _scripts(), "packaging no longer exposes a CLI"


@pytest.mark.parametrize("name,target", sorted(_scripts().items()))
def test_every_console_script_target_resolves(name: str, target: str):
    """Regression: this was declared as `sv_engine:main`, and `sv_engine` has
    no `main`, so `sv-engine --help` died with an ImportError."""
    module_path, _, attribute = target.partition(":")
    module = importlib.import_module(module_path)

    assert hasattr(module, attribute), f"{target} does not exist"
    assert callable(getattr(module, attribute)), f"{target} is not callable"


def test_the_console_script_reaches_the_real_cli():
    from sv_engine.cli import build_parser

    module_path, _, attribute = _scripts()["sv-engine"].partition(":")
    entry = getattr(importlib.import_module(module_path), attribute)

    with pytest.raises(SystemExit) as exit_info:
        entry(["--help"])
    assert exit_info.value.code == 0
    # The parser it reaches is the one carrying the documented subcommands.
    assert {"index", "search", "serve", "recover", "eval"} <= set(
        build_parser()._subparsers._group_actions[0].choices
    )
