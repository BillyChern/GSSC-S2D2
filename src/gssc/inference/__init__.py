"""Evaluation, prediction generation, BEV second-task evaluator, D4 TTA.

Public entry points (intended for visitors writing custom scripts):

>>> from gssc.inference import run_evaluation        # 3D SSC headline path
>>> from gssc.inference import evaluate_bev          # 2D BEV second task

Driver scripts ``scripts/eval.py`` and ``scripts/infer.py`` are the
recommended way to invoke these from the command line.
"""
from __future__ import annotations

from gssc.inference.evaluate import run_evaluation
from gssc.inference.evaluate_bev import evaluate_bev

__all__ = [
    "evaluate_bev",
    "run_evaluation",
]
