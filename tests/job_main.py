"""`lamino-pal job <dir>` with the ops of jobops.py registered."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jobops  # noqa: E402

from lamino_pal import jobs  # noqa: E402
from lamino_pal.cli import main  # noqa: E402
from lamino_pal.ops import REGISTRY  # noqa: E402

REGISTRY.update({o.name: o for o in jobops.OPS})
jobs.CANCEL_GRACE_S = 1.0
sys.exit(main())
