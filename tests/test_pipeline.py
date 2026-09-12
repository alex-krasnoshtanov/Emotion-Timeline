"""The timeline: scene grouping, the aggregation, and the record it publishes.

Two claims here are worth more than the rest. The first is that a scene's
probability is weighted by *duration*, because unweighted averaging is the
obvious implementation and it lets a run of two-word interjections outvote the
paragraph they interrupt. The second is that the record's scenes have to regroup
out of the transcript it names -- the record carries no text, so without that
check a timeline could quietly describe a different recording.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from emotion_timeline.data.labels import EMOTIONS
from emotion_timeline.pipeline import timeline as pipeline

ROOT = Path(__file__).resolve().parents[1]


def committed() -> pipeline.Timeline:
    return pipeline.Timeline.load()


def transcript() -> list[pipeline.Segment]:
    return pipeline.read_segments(pipeline.DEFAULT_SEGMENTS)


def segments(*spans: tuple[float, float]) -> list[pipeline.Segment]:
    return [
        pipeline.Segment(start, end, f"line {index}") for index, (start, end) in enumerate(spans)
    ]


def flat(rows: int, peak: str) -> np.ndarray:
    """A probability matrix where every row prefers ``peak``."""
    values = np.full((rows, len(EMOTIONS)), 0.1)
    values[:, EMOTIONS.index(peak)] = 0.4
    normalised: np.ndarray = values / values.sum(axis=1, keepdims=True)
    return normalised


# --- scene grouping -----------------------------------------------------------


def test_a_long_silence_starts_a_new_scene_and_a_short_one_does_not() -> None:
    scenes = pipeline.group(segments((0, 5), (5.5, 8), (20, 25)), gap=1.0)
    assert [len(scene.segments) for scene in scenes] == [2, 1]
    assert scenes[0].start_s == 0 and scenes[0].end_s == 8
    assert scenes[1].index == 1


def test_the_gap_is_a_threshold_not_a_rule_about_silence() -> None:
    """Same transcript, two thresholds, two different timelines."""
    spans = segments((0, 5), (7, 9), (11, 13))
    assert len(pipeline.group(spans, gap=1.0)) == 3
    assert len(pipeline.group(spans, gap=5.0)) == 1


def test_grouping_nothing_gives_nothing() -> None:
    assert pipeline.group([]) == []


def test_a_scene_joins_its_segments_text_and_skips_the_empty_ones() -> None:
    scene = pipeline.group(
        [
            pipeline.Segment(0, 1, "first"),
            pipeline.Segment(1, 2, ""),
            pipeline.Segment(2, 3, "last"),
        ]
    )[0]
    assert scene.text == "first last"


# --- the transcript reader ----------------------------------------------------


def test_the_committed_transcript_reads_as_segments() -> None:
    parts = transcript()
    assert len(parts) == 316
    assert parts[0].start_s == 1.915
    assert all(segment.end_s >= segment.start_s for segment in parts)


def test_either_text_column_is_accepted(tmp_path: Path) -> None:
    """`hypothesis` from the annotated benchmark, `text` from `transcribe`."""
    for column in ("text", "hypothesis"):
        path = tmp_path / f"{column}.csv"
        path.write_text(f"start_s,end_s,{column}\n0,1,hello\n", encoding="utf-8", newline="\n")
        assert pipeline.read_segments(path)[0].text == "hello"


def test_a_csv_with_no_text_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("start_s,end_s,words\n0,1,hello\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="no text column"):
        pipeline.read_segments(path)


def test_an_empty_csv_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("start_s,end_s,text\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="no rows"):
        pipeline.read_segments(path)


def test_paths_are_recorded_relative_to_the_repository() -> None:
    assert pipeline.relative(pipeline.DEFAULT_SEGMENTS) == "benchmarks/stt/assemblyai-best.csv"
    outside = Path.home() / "elsewhere.csv"
    assert pipeline.relative(outside) == outside.resolve().as_posix()


# --- aggregation --------------------------------------------------------------


def test_a_scenes_probability_is_weighted_by_duration_not_by_chunk_count() -> None:
    """The long chunk wins, which unweighted averaging would get backwards."""
    scene = pipeline.group(segments((0, 60), (60, 61), (61, 62)), gap=1.0)[0]
    pieces = pipeline.chunks([scene], max_chars=6)
    assert [round(piece.seconds) for piece in pieces] == [60, 1, 1]
    values = np.vstack([flat(1, "Sadness"), flat(2, "Joy")])
    aggregated = pipeline.aggregate(pieces, values, 1)
    assert EMOTIONS[int(aggregated.argmax())] == "Sadness"
    assert EMOTIONS[int(values.mean(axis=0).argmax())] == "Joy"


def test_a_scene_of_zero_length_segments_averages_rather_than_dividing_by_zero() -> None:
    scene = pipeline.group([pipeline.Segment(0, 0, "a"), pipeline.Segment(0, 0, "b")], gap=1.0)[0]
    pieces = pipeline.chunks([scene], max_chars=1)
    aggregated = pipeline.aggregate(pieces, np.vstack([flat(1, "Fear"), flat(1, "Fear")]), 1)
    assert EMOTIONS[int(aggregated.argmax())] == "Fear"


def test_each_scene_keeps_its_own_chunks() -> None:
    scenes = pipeline.group(segments((0, 1), (5, 6)), gap=1.0)
    pieces = pipeline.chunks(scenes)
    aggregated = pipeline.aggregate(pieces, np.vstack([flat(1, "Anger"), flat(1, "Joy")]), 2)
    assert [EMOTIONS[int(row.argmax())] for row in aggregated] == ["Anger", "Joy"]


# --- the record ---------------------------------------------------------------


def built() -> dict[str, Any]:
    scenes = pipeline.group(segments((0, 5), (5.5, 8), (20, 25)), gap=1.0)
    pieces = pipeline.chunks(scenes, max_chars=6)
    assert [piece.scene for piece in pieces] == [0, 0, 1]
    return pipeline.build_record(
        scenes,
        pieces,
        np.vstack([flat(2, "Joy"), flat(1, "Fear")]),
        np.vstack([flat(2, "Joy"), flat(1, "Anger")]),
        source="somewhere.csv",
        gap=1.0,
        primary_model={"name": "B", "model": "b", "temperature": 1.2},
        second_model={"name": "A", "model": "a", "temperature": 2.6},
    )


def test_the_record_marks_agreement_scene_by_scene() -> None:
    record = built()
    rows = record["timeline"]
    assert [row["emotion"] for row in rows] == ["Joy", "Fear"]
    assert [row["second_opinion"] for row in rows] == ["Joy", "Anger"]
    assert [row["agreed"] for row in rows] == [True, False]
    assert record["agreement"]["scenes"] == 1
    assert record["agreement"]["share"] == 0.5


def test_the_record_does_not_repeat_the_transcript() -> None:
    """Scene text lives in the transcript; duplicating it lets the two drift."""
    assert "text" not in json.dumps(built()["timeline"])


# --- the committed timeline ---------------------------------------------------


def test_the_committed_timeline_holds_together() -> None:
    assert pipeline.check_consistency(committed(), transcript()) == []


def test_the_committed_timeline_is_the_documentary() -> None:
    report = committed()
    assert report.raw["segments"] == 316
    assert report.raw["scenes"] == 47
    assert report.gap == 1.0
    assert round(report.duration / 60) == 52
    assert report.raw["source"] == "benchmarks/stt/assemblyai-best.csv"


def test_the_emotion_is_the_native_model_and_the_translation_is_the_second_opinion() -> None:
    """No combination rule beat B alone, so the timeline does not use one."""
    report = committed()
    assert "ruBERT" in report.raw["primary"]["name"]
    assert "translate" in report.raw["second_opinion"]["name"]
    assert (
        report.raw["primary"]["held_out_accuracy"]
        > report.raw["second_opinion"]["held_out_accuracy"]
    )


def test_documentary_narration_comes_out_mostly_neutral() -> None:
    """The headline of the chapter, and the reason the confidence strip matters."""
    counts = committed().counts()
    assert counts["Neutral"] == 26
    assert sum(counts.values()) == 47
    assert set(counts) == set(EMOTIONS)


def test_the_agreement_rate_carries_its_caveat() -> None:
    """It is a consistency signal. The record has to say so, beside the number."""
    agreement = committed().agreement
    assert agreement["scenes"] == 17
    assert "not an accuracy" in agreement["caveat"]
    assert "ru-izard" in agreement["measured_on"]


# --- what the consistency check catches ---------------------------------------


def broken(**changes: Any) -> pipeline.Timeline:
    """The committed record with one scene altered."""
    report = committed()
    raw = json.loads(json.dumps(report.raw))
    raw["timeline"][0].update(changes)
    return pipeline.Timeline(raw=raw, source=report.source, digest=report.digest)


def test_a_flag_that_contradicts_its_own_two_predictions_is_caught() -> None:
    problems = pipeline.check_consistency(broken(agreed=not committed().scenes[0]["agreed"]))
    assert any("contradicts" in problem for problem in problems)


def test_an_emotion_outside_the_seven_is_caught() -> None:
    assert any(
        "not one of the seven" in p for p in pipeline.check_consistency(broken(emotion="Love"))
    )


def test_a_confidence_below_one_seventh_is_not_a_top_class_probability() -> None:
    assert any(
        "top-class probability" in p for p in pipeline.check_consistency(broken(confidence=0.05))
    )


def test_a_scene_that_ends_before_it_starts_is_caught() -> None:
    assert any("ends before it starts" in p for p in pipeline.check_consistency(broken(end_s=0.0)))


def test_overlapping_scenes_are_caught() -> None:
    report = committed()
    raw = json.loads(json.dumps(report.raw))
    raw["timeline"][1]["start_s"] = raw["timeline"][0]["start_s"] - 1
    problems = pipeline.check_consistency(
        pipeline.Timeline(raw=raw, source=report.source, digest=report.digest)
    )
    assert any("before the scene before it ended" in problem for problem in problems)


def test_headers_that_disagree_with_the_list_are_caught() -> None:
    report = committed()
    raw = json.loads(json.dumps(report.raw))
    raw["scenes"] = 3
    raw["segments"] = 9
    raw["duration_s"] = 12.0
    raw["agreement"] = {**raw["agreement"], "scenes": 1, "share": 0.5}
    problems = pipeline.check_consistency(
        pipeline.Timeline(raw=raw, source=report.source, digest=report.digest)
    )
    assert len(problems) == 5


def test_an_empty_record_is_refused() -> None:
    report = committed()
    empty = pipeline.Timeline(raw={**report.raw, "timeline": []}, source=report.source, digest="")
    assert pipeline.check_consistency(empty) == ["the record carries no scenes"]


def test_a_record_whose_scenes_do_not_regroup_out_of_the_transcript_is_caught() -> None:
    """The check that stops a timeline describing a recording it was not built from."""
    report = committed()
    raw = {**json.loads(json.dumps(report.raw)), "gap_seconds": 5.0}
    problems = pipeline.check_consistency(
        pipeline.Timeline(raw=raw, source=report.source, digest=report.digest), transcript()
    )
    assert any("regroups into" in problem for problem in problems)


def test_a_scene_starting_somewhere_the_transcript_does_not_is_caught() -> None:
    problems = pipeline.check_consistency(broken(start_s=7.0), transcript())
    assert any("does not start where the transcript does" in problem for problem in problems)


# --- the table and the CSV ----------------------------------------------------


def test_the_table_joins_the_record_back_to_the_transcript() -> None:
    rows = pipeline.table(committed(), transcript())
    assert len(rows) == 47
    assert rows[0]["text"].startswith("Козацкая")
    assert all(row["text"] for row in rows)


def test_the_table_refuses_a_transcript_that_does_not_match() -> None:
    with pytest.raises(ValueError, match="against 47 in the record"):
        pipeline.table(committed(), transcript()[:10])


def test_the_csv_carries_both_predictions_and_both_confidences(tmp_path: Path) -> None:
    written = pipeline.write_csv(pipeline.table(committed(), transcript()), tmp_path / "t.csv")
    header = written.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(pipeline.CSV_COLUMNS)
    for column in ("emotion", "confidence", "second_opinion", "second_confidence", "agreed"):
        assert column in header


def test_the_committed_csv_matches_the_committed_record(tmp_path: Path) -> None:
    fresh = pipeline.write_csv(pipeline.table(committed(), transcript()), tmp_path / "t.csv")
    assert fresh.read_bytes() == (ROOT / "benchmarks" / "pipeline" / "timeline.csv").read_bytes()


def test_describe_names_the_two_models_and_the_agreement() -> None:
    out = "\n".join(pipeline.describe(committed()))
    assert "47 scenes" in out and "1s silence gap" in out
    assert "ruBERT" in out and "translate" in out
    assert "agree on 17 of 47" in out
    assert "not an accuracy" in out


# --- what the chapter and the README publish ---------------------------------


def test_the_second_opinion_leans_disgust() -> None:
    """A says Disgust sixteen times where B says it three. The chapter says so."""
    scenes = committed().scenes
    assert sum(1 for scene in scenes if scene["second_opinion"] == "Disgust") == 24
    assert sum(1 for scene in scenes if scene["emotion"] == "Disgust") == 4


def test_agreement_and_confidence_move_together() -> None:
    """Two signals never fitted to each other, pointing the same way."""
    scenes = committed().scenes
    agreed = [float(s["confidence"]) for s in scenes if s["agreed"]]
    split = [float(s["confidence"]) for s in scenes if not s["agreed"]]
    assert round(sum(agreed) / len(agreed), 3) == 0.422
    assert round(sum(split) / len(split), 3) == 0.384


def test_the_confidences_are_low_and_the_chapter_admits_it() -> None:
    values = sorted(float(scene["confidence"]) for scene in committed().scenes)
    median = values[len(values) // 2]
    assert round(median, 3) == 0.382
    assert "0.382" in (ROOT / "docs" / "pipeline.md").read_text(encoding="utf-8")


def test_the_loudest_scene_is_agreed_and_so_are_the_two_quietest() -> None:
    """The claim the chapter closes its results on: agreement is not confidence."""
    speaking = [s for s in committed().scenes if s["emotion"] != "Neutral"]
    ranked = sorted(speaking, key=lambda s: float(s["confidence"]))
    loudest = ranked[-1]
    assert (loudest["emotion"], float(loudest["confidence"])) == ("Joy", 0.7691)
    assert loudest["agreed"]
    assert [float(s["confidence"]) for s in ranked[:2]] == [0.2297, 0.2370]
    assert all(s["agreed"] for s in ranked[:2])


def test_the_readme_quotes_the_record_it_was_built_from() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = committed()
    assert f"| Segments | {report.raw['segments']} |" in readme
    assert f"| Scenes, at a 1s silence gap | {report.raw['scenes']} |" in readme
    assert f"{report.agreement['scenes']} ({float(report.agreement['share']):.1%})" in readme
    # The line that must never be dropped: this is not an accuracy.
    assert "None of this is an accuracy" in readme


def test_the_chapter_ends_by_saying_what_it_does_not_establish() -> None:
    chapter = (ROOT / "docs" / "pipeline.md").read_text(encoding="utf-8")
    assert "## What this does not establish" in chapter
    assert "There are no labels on this recording" in chapter


# --- chunking -----------------------------------------------------------------


def test_chunks_pack_whole_segments_up_to_the_budget() -> None:
    scene = pipeline.group(segments((0, 5), (5, 10), (10, 15)), gap=1.0)[0]
    pieces = pipeline.chunks([scene], max_chars=14)
    assert [piece.text for piece in pieces] == ["line 0 line 1", "line 2"]
    assert [piece.seconds for piece in pieces] == [10.0, 5.0]


def test_chunks_never_run_across_a_scene_boundary() -> None:
    scenes = pipeline.group(segments((0, 1), (9, 10)), gap=1.0)
    assert [piece.scene for piece in pipeline.chunks(scenes, max_chars=400)] == [0, 1]


def test_a_segment_too_long_to_read_is_split_between_sentences() -> None:
    """The residual transcriber dependency: a 974-character paragraph."""
    long = pipeline.Segment(0, 10, "Aaa bbb ccc. Ddd eee fff. Ggg hhh iii.")
    pieces = pipeline.chunks([pipeline.Scene(0, (long,))], max_chars=26)
    assert [piece.text for piece in pieces] == ["Aaa bbb ccc. Ddd eee fff.", "Ggg hhh iii."]
    # Time is apportioned by characters and still adds up to the segment's.
    assert round(sum(piece.seconds for piece in pieces), 6) == 10.0


def test_a_sentence_longer_than_the_budget_is_cut_rather_than_dropped() -> None:
    assert pipeline.sentences("x" * 25, max_chars=10) == ["x" * 10, "x" * 10, "x" * 5]


def test_no_chunk_of_either_committed_transcript_exceeds_the_budget() -> None:
    """The claim the 400 was chosen for: neither model ever truncates."""
    for path, gap in (
        (pipeline.DEFAULT_SEGMENTS, 1.0),
        (ROOT / "benchmarks" / "pipeline" / "whisper-turbo.csv", 2.0),
    ):
        pieces = pipeline.chunks(pipeline.group(pipeline.read_segments(path), gap))
        assert pieces
        assert max(len(piece.text) for piece in pieces) <= pipeline.CHUNK_CHARS


def test_the_committed_record_says_what_unit_it_classified() -> None:
    assert committed().raw["chunks"] == 142
    assert committed().raw["chunk_chars"] == 400


def test_aggregating_probabilities_that_do_not_cover_every_chunk_is_refused() -> None:
    scenes = pipeline.group(segments((0, 1), (5, 6)), gap=1.0)
    pieces = pipeline.chunks(scenes)
    with pytest.raises(ValueError, match="against 2 chunks"):
        pipeline.aggregate(pieces, flat(1, "Joy"), len(scenes))


# --- how much of a timeline is the transcriber --------------------------------


def whisper() -> pipeline.Timeline:
    return pipeline.Timeline.load(ROOT / "benchmarks" / "pipeline" / "timeline-whisper.json")


def test_the_second_transcript_is_the_same_recording() -> None:
    other = whisper()
    assert other.raw["segments"] == 947
    assert other.raw["scenes"] == 54
    assert round(other.duration / 60) == 52
    assert (
        pipeline.check_consistency(
            other, pipeline.read_segments(ROOT / "benchmarks" / "pipeline" / "whisper-turbo.csv")
        )
        == []
    )


def test_changing_the_transcriber_moves_more_than_a_third_of_the_timeline() -> None:
    """The finding, and the ceiling on every other claim in the chapter."""
    overlap = pipeline.runtime_agreement(committed(), whisper())
    assert overlap["covered"] == 2740
    assert overlap["share"] == 0.6197
    assert next(iter(overlap["disagreements"])) == "Neutral -> Fear"


def test_runtime_agreement_reports_the_coverage_it_measured_over() -> None:
    """A share over the seconds two timelines happen to share means nothing alone."""
    overlap = pipeline.runtime_agreement(committed(), whisper())
    assert 0.0 < overlap["coverage"] < 1.0
    assert overlap["covered"] < overlap["grid_points"]


def test_a_timeline_agrees_with_itself_everywhere() -> None:
    overlap = pipeline.runtime_agreement(committed(), committed())
    assert overlap["share"] == 1.0
    assert overlap["disagreements"] == {}


def test_a_moment_in_no_scene_has_no_emotion() -> None:
    """Silence longer than the threshold belongs to nobody, and is not guessed at."""
    assert pipeline.emotion_at(committed(), 0.0) is None
    assert pipeline.emotion_at(committed(), 3.0) == "Neutral"


# --- the optional front end ---------------------------------------------------


def test_a_transcription_round_trips_through_the_three_columns(tmp_path: Path) -> None:
    """The seam: whatever writes these three columns can feed the timeline."""
    from emotion_timeline.pipeline import transcribe

    original = [
        pipeline.Segment(0.0, 1.25, "первый"),
        pipeline.Segment(3.5, 9.0, "second, with a comma"),
    ]
    written = transcribe.write_segments(original, tmp_path / "segments.csv")
    assert bytes([13]) not in written.read_bytes()  # LF, not CRLF
    assert pipeline.read_segments(written) == original


def test_the_transcriber_is_turbo_and_says_why() -> None:
    from emotion_timeline.pipeline import transcribe

    assert transcribe.MODEL == "large-v3-turbo"
    assert "7.75%" in (transcribe.__doc__ or "")


# --- the figure ---------------------------------------------------------------


def test_the_figure_renders_and_is_stamped_with_the_record(tmp_path: Path) -> None:
    from emotion_timeline import figures as shared
    from emotion_timeline.pipeline import figures as pipeline_figures

    written = pipeline_figures.render_all(committed(), tmp_path)
    assert len(written) == len(pipeline_figures.FIGURES)
    assert shared.read_stamp(written[0]) == committed().digest


def test_the_committed_figure_is_current() -> None:
    from emotion_timeline.pipeline import figures as pipeline_figures

    assert pipeline_figures.check_figures_current(committed(), ROOT / "assets") == []


def test_every_emotion_has_a_colour() -> None:
    from emotion_timeline import figures as shared

    assert set(shared.EMOTION_COLOURS) == set(EMOTIONS)
