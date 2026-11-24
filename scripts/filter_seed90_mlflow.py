import sqlite3
from pathlib import Path

base = Path("analysis_inputs/duet1p0")

source = base / (
    "20261124_005539_noops_correction_run_step180_to_step300_"
    "resuming_from_20261123_004046_noops_run.db"
)

destination = base / "seed90_corrected_segment2_canonical.db"

canonical_run = "4519e2f60ca84b7c9bfc7301e4217c38"

if destination.exists():
    raise RuntimeError(
        f"Destination already exists: {destination}"
    )

with sqlite3.connect(source) as original:
    with sqlite3.connect(destination) as copy:
        original.backup(copy)
        copy.execute(
            "DELETE FROM metrics WHERE run_uuid != ?",
            (canonical_run,)
        )
        copy.commit()

with sqlite3.connect(destination) as db:
    rows = db.execute("""
        SELECT MIN(step), MAX(step), COUNT(DISTINCT step)
        FROM metrics
        WHERE key = 'duet/generated_response_tokens'
    """).fetchone()

print("Canonical MLflow segment 2")
print("Run ID:", canonical_run)
print("First step:", rows[0])
print("Last step:", rows[1])
print("Token-metric steps:", rows[2])

assert rows[0] == 181
assert rows[1] == 289
assert rows[2] == 109

print("PASS: Canonical metrics isolated; original database unchanged.")
