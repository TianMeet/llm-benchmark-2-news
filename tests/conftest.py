import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 共享 fixture ──────────────────────────────────────────────────────

SCORING_PROFILE_PATH = (
    ROOT / "configs" / "eval_contract" / "scoring_profile.template.yaml"
)


@pytest.fixture
def scoring_profile_path() -> Path:
    """返回真实 scoring_profile.template.yaml 的路径。"""
    return SCORING_PROFILE_PATH
