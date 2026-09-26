import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from duet.scheduler.core import DifficultyOnlyScheduler


@pytest.fixture
def difficulty_scheduler() -> DifficultyOnlyScheduler:
    return DifficultyOnlyScheduler()
