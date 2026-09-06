"""Speech-to-text adapters and the word-error-rate harness."""

from emotion_timeline.stt.wer import WerResult, compare, in_window, load, score

__all__ = ["WerResult", "compare", "in_window", "load", "score"]
