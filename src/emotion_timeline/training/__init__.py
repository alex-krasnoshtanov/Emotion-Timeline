"""Fine-tuning a classifier on the rebuilt dataset, and recording what it scored."""

from emotion_timeline.training import evaluate, preflight, run, splits, summary

__all__ = ["evaluate", "preflight", "run", "splits", "summary"]
