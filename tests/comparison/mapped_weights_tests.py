"""Bootstrap AIMDO before Torch and run focused tests in a separate process."""

import sys

import pytest

from unitunner.execution.allocator_bootstrap import initialize_vbar

initialize_vbar()


raise SystemExit(pytest.main(["-q", *sys.argv[1:]]))
