"""The registration engine: register a time series, one position at a time.

`core` builds a run from its settings and runs its steps (init, flag, finalize); the
public names are re-exported here, so `from biahub.registration.engine import init_run`
works wherever they live.
"""

from biahub.registration.engine.cluster import estimate_transform_series, run_timepoint_jobs
from biahub.registration.engine.core import (
    MAX_MINUTES,
    block_failure,
    build_estimator,
    finalize_run,
    flag_run,
    init_run,
    transforms_for_file,
)
from biahub.registration.engine.disk import JOB_FAILED_PREFIX, RunJournal
from biahub.registration.engine.fallback import (
    PassResult,
    flag_timepoints,
    neighbour_consensus_initial_candidates,
    repair,
    select_flagged,
)
from biahub.registration.engine.series import (
    SeriesResult,
    estimate_propagated,
    estimate_series,
)
from biahub.registration.methods.landmark.bead_matching import build_beads_estimator

__all__ = [
    "JOB_FAILED_PREFIX",
    "MAX_MINUTES",
    "PassResult",
    "RunJournal",
    "SeriesResult",
    "block_failure",
    "build_beads_estimator",
    "build_estimator",
    "estimate_propagated",
    "estimate_series",
    "estimate_transform_series",
    "finalize_run",
    "flag_run",
    "flag_timepoints",
    "init_run",
    "neighbour_consensus_initial_candidates",
    "repair",
    "run_timepoint_jobs",
    "select_flagged",
    "transforms_for_file",
]
