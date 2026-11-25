import importlib.util
import sys
from pathlib import Path

script = Path("scripts/duet_six_runs_offline.py")

spec = importlib.util.spec_from_file_location(
    "duet_six_runs_corrected", script
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.MANIFEST[("noop", 90)] = [
    (
        "duet1p0",
        "20261123_004046_noops_run_300_steps_seed_90_correction_run.db",
        1, 180
    ),
    (
        "duet1p0",
        "seed90_corrected_segment2_canonical.db",
        181, 280
    ),
    (
        "duet1p0",
        "20261124_132909_noops_correction_run_step280_to_step_300_resuming_from_20261124_005539_noops_correction_run.db",
        281, 300
    ),
]

sys.argv = [
    str(script),
    "--inputs", "analysis_inputs",
    "--out", "analysis_outputs/six_runs_seed90_corrected",
]

module.main()
