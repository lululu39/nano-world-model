#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from latent_world.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
