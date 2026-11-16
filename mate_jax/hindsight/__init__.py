"""Hindsight -> Foresight: distilling clairvoyant, rate-limited camera schedules into decentralized policies."""

from mate_jax.hindsight.student import HindsightDistiller, LocalWaypoints, PriorActor, StudentFeatures, trajectory_planner
from mate_jax.hindsight.teacher import MOVES, NUM_MOVES, HindsightTeacher, TeacherPlan, move_action, record_targets


__all__ = [
    'MOVES', 'NUM_MOVES', 'HindsightTeacher', 'TeacherPlan', 'move_action', 'record_targets',
    'HindsightDistiller', 'LocalWaypoints', 'PriorActor', 'StudentFeatures', 'trajectory_planner',
]
