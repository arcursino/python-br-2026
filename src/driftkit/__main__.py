"""Permite `python -m driftkit`, além do entry point `driftkit`.

O entry point instalado depende do PATH. No Colab, `python -m driftkit`
usa o MESMO interpretador do notebook — mais robusto e sem ambiguidade
sobre qual ambiente está sendo executado.
"""

from __future__ import annotations

import sys

from .cli import app

if __name__ == "__main__":
    sys.exit(app())
