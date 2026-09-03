"""The public inference package must not depend on training or figure code.

This repo is the PUBLIC inference API. Training scripts and the paper's figure production moved to
``paper-ephys-atlas``, which depends one-way on the installed ``ephysatlas`` package. What stays
here has a one-way dependency rule of its own:

    src/ephysatlas/     PUBLIC: inference + feature computation (the installed package)
    examples/           public inference examples (repo root, not installed; import *from* the package)

``src/ephysatlas`` is what people ``pip install`` and import. The dependency must never flow back:
importing the public surface must pull in nothing under ``examples/`` (nor a re-introduced
``training/``). This test is the tripwire -- it passes today and fails the day a public module grows
an ``import training...`` / figure / example import, before that leaks into a release.

Run in a **subprocess** on purpose: the pytest process that runs the rest of the suite imports the
torch-heavy modules (``test_unit_encoder``, the ``*_release`` tests), so this process's
``sys.modules`` is already polluted. A clean interpreter is the only way to see what the public
surface *alone* pulls in.
"""

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The public surface a user touches: the top-level package + its re-exported entry point, the
# model registry and wrappers (the load path), and the feature/data/aggregation/anatomy stack.
_PUBLIC_IMPORTS = [
    "import ephysatlas",
    "from ephysatlas import load_pretrained, model_registry",
    "import ephysatlas.models",
    "import ephysatlas.features",
    "import ephysatlas.data",
    "import ephysatlas.aggregation",
    "import ephysatlas.anatomy",
]


def _probe_public_surface() -> dict:
    """Import the public surface in a clean interpreter and report what it loaded.

    Returns a dict with ``offenders`` (modules whose file lives under ``training/`` or
    ``examples/``) and ``heavy`` (whether xgboost / torch were imported at module scope).
    """
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        {imports}

        repo = Path(sys.argv[1]).resolve()
        watched = (repo / "training", repo / "examples")
        offenders = []
        for name, module in list(sys.modules.items()):
            file = getattr(module, "__file__", None)
            if not file:
                continue
            try:
                path = Path(file).resolve()
            except (OSError, ValueError):
                continue
            if any(str(path).startswith(str(d)) for d in watched):
                offenders.append({{"module": name, "file": str(path)}})

        import json
        # xgboost and torch segfault together on macOS arm64, so the public surface imports both
        # lazily (inside methods). Report whether either leaked to import time -- a regression here
        # is what forces the rest of the suite into separate processes.
        print(json.dumps({{
            "offenders": offenders,
            "heavy": sorted(m for m in ("xgboost", "torch") if m in sys.modules),
        }}))
        """
    ).format(imports="\n".join(_PUBLIC_IMPORTS))

    result = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"probe failed:\n{result.stderr[-3000:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestRepoSegregation(unittest.TestCase):
    def test_public_surface_imports_nothing_from_training_or_examples(self):
        report = _probe_public_surface()
        self.assertEqual(
            report["offenders"],
            [],
            msg=(
                "the public package src/ephysatlas transitively imported training/ or examples/ "
                f"code, which breaks the one-way dependency rule: {report['offenders']}"
            ),
        )

    def test_public_surface_pulls_no_torch_or_xgboost_at_import(self):
        # Not strictly segregation, but the same boundary: the installed package stays lightweight,
        # importing its heavy runtimes only when a model that needs them is actually used.
        report = _probe_public_surface()
        self.assertEqual(report["heavy"], [])


if __name__ == "__main__":
    unittest.main()
