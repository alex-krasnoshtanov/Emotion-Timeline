"""The model-selection audit: the arithmetic, and the claim that does not survive it.

Neither log can be recomputed. The runs are gone, and the dataset behind the
earliest of them is client material that cannot be republished. So what is
tested here is everything the two records can be made to say about each other,
plus the one derivation that gets at what neither of them recorded: the size of
the evaluation set each run was scored over.

The mistake pinned in this file is the inherited conclusion that a
feature-engineered network beat a transformer. It is what you get by sorting the
run log and reading off the top; the tests assert both that sorting gives that
answer and that the answer is not a comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_timeline.data import build as ds
from emotion_timeline.selection import figures as sel_figures
from emotion_timeline.selection import runs as sel

# --- the committed records ---------------------------------------------------


def test_both_records_are_internally_consistent() -> None:
    assert sel.check_consistency(sel.SelectionReport.load()) == []


def test_the_logs_are_the_size_the_docs_say() -> None:
    report = sel.SelectionReport.load()
    assert len(report.runs) == 101
    assert len(report.submitted) == 8
    assert report.dataset == "GoEmotions (simplified) - 7 emotions"
    # No test split. Every submitted metric is a validation score on the same
    # 15% the runs were tuned against, which is why none of them is quoted in
    # the README as the performance of anything.
    assert report.split == "85/15 train/val"


def test_the_published_audit_figures() -> None:
    """Every number the README and docs/model-selection.md quote, in one place."""
    audit = sel.audit(sel.SelectionReport.load())
    assert audit == {
        "submitted_rows": 8,
        "logged_runs": 101,
        "measurable_runs": 79,
        "distinct_divisors": 27,
        "collapsed_runs": 17,
        "best_overall": "pytorch_mlp_gpu",
        "best_overall_f1_macro": pytest.approx(0.8219, abs=5e-5),
        "best_transformer": "distilbert",
        "best_transformer_f1_macro": pytest.approx(0.7515, abs=5e-5),
        "shared_evaluation_size": 6_044_800,
        "submitted_winner_weighted": "distilbert",
        "submitted_winner_macro": "distilbert",
        "largest_swing_places": 3,
        "smallest_gap_to_a_logged_run": pytest.approx(0.000206, abs=5e-7),
        "nearest_of_another_model_type": 3,
        "neutral_caps": [6000, 10000],
    }


# --- recovering what neither log wrote down ----------------------------------


@pytest.mark.parametrize(
    ("recorded", "divisor"),
    [
        ("0.508928571", 112),
        ("0.8653125", 3200),
        ("0.78824775", 1889),
        ("0.440860215", 93),
        ("0.714088398", 724),
    ],
)
def test_an_accuracy_recovers_a_divisor_of_its_evaluation_set(recorded: str, divisor: int) -> None:
    """An accuracy is whole correct over whole total, so its denominator survives."""
    assert sel.evaluation_divisor(recorded) == divisor


@pytest.mark.parametrize("recorded", ["0.2796", "0.5", "0.75", "0.8625", "0.7143"])
def test_a_rounded_accuracy_recovers_nothing(recorded: str) -> None:
    """Four decimal places is not imprecision, it is unrecoverability.

    0.2796 reads as 699/2500 in lowest terms and 2,500 is not a set anybody
    scored. 0.8625 is exactly 69/80 and may well have come from 2,760 of 3,200 —
    which is the point: at this precision the fraction cannot be trusted, so the
    audit declines rather than guessing. 22 of the 101 runs are in this state.
    """
    assert sel.evaluation_divisor(recorded) is None


def test_an_accuracy_that_is_not_a_plausible_fraction_recovers_nothing() -> None:
    """Precision alone is not enough; the fraction has to actually reproduce it."""
    assert sel.evaluation_divisor("0.123456789") is None


def test_the_log_rounded_22_of_its_101_rows_past_recovery() -> None:
    report = sel.SelectionReport.load()
    assert len(report.measurable) == 79
    assert len(report.runs) - len(report.measurable) == 22
    assert all(run.divisor is not None for run in report.measurable)


def test_a_run_with_no_recoverable_divisor_gets_no_interval() -> None:
    report = sel.SelectionReport.load()
    rounded = [run for run in report.runs if run.divisor is None]
    assert rounded and all(run.interval is None for run in rounded)
    assert all(run.interval is not None for run in report.measurable)


def test_a_wilson_interval_tightens_as_the_evaluation_set_grows() -> None:
    narrow = sel.wilson(80, 100)
    wide = sel.wilson(8, 10)
    assert narrow[0] > wide[0] and narrow[1] < wide[1]
    assert 0.0 < wide[0] < 0.8 < wide[1] < 1.0


# --- the one thing here that is recomputed rather than cross-checked ---------


def test_a_collapsed_models_scores_follow_from_its_accuracy_alone() -> None:
    """Predict one class for everything and both F1 scores are determined.

    The majority class scores 2k/(n+k) and the other six score zero, so macro
    averaging divides by seven and support weighting multiplies by k/n. No
    property of the model appears in it, which is what makes it a test rather
    than a description.
    """
    macro, weighted = sel.single_class_scores(57, 112)
    assert macro == pytest.approx(0.096365173, abs=5e-10)
    assert weighted == pytest.approx(0.34330093, abs=5e-9)

    macro, weighted = sel.single_class_scores(17, 93)
    assert macro == pytest.approx(0.044155844, abs=5e-10)
    assert weighted == pytest.approx(0.056500489, abs=5e-10)


def test_seventeen_of_the_runs_learned_nothing_at_all() -> None:
    """And four different architectures agreed on it to nine decimal places.

    Twelve of the seventeen share one triple of scores exactly — 0.096365173
    macro, 0.34330093 weighted, 57 of 112 right — across naive Bayes, an LSTM,
    an RNN and XLM-RoBERTa. Four architectures cannot agree to nine decimals
    unless they made identical predictions, and the closed form says which
    predictions those were.
    """
    report = sel.SelectionReport.load()
    collapsed = report.collapsed()
    assert len(collapsed) == 17
    assert {run.model_name for run in collapsed} == {
        "naive_bayes",
        "lstm",
        "rnn",
        "xlm_roberta",
        "birnn_optimized_direct",
    }
    # All three model types are represented, so this is not one broken family.
    assert {sel.model_type_of(run.model_type) for run in collapsed} == {
        "traditional_ml",
        "deep_learning",
        "transformer",
    }
    majority = [run for run in collapsed if run.divisor == 112]
    assert len(majority) == 12
    assert {run.f1_macro for run in majority} == {0.096365173}


def test_the_class_count_is_the_studys_own() -> None:
    """The closed form divides by seven because the problem has seven classes."""
    from emotion_timeline.data import labels

    assert sel.CLASSES == len(labels.EMOTIONS) == 7


def test_a_model_that_learned_something_does_not_match_the_closed_form() -> None:
    report = sel.SelectionReport.load()
    best, _ = report.headline_pair()
    assert best not in report.collapsed()
    macro, _ = sel.single_class_scores(round(best.accuracy * 3200), 3200)
    assert abs(macro - best.f1_macro) > 0.5


# --- the claim that does not survive -----------------------------------------


def test_sorting_the_log_reproduces_the_inherited_claim() -> None:
    """First, that the claim is really what the log says if you just sort it.

    The coursework's conclusion — a feature-engineered network beating a
    transformer — is what reading the top of the table gives you. Pinning it
    here means the next test is refuting the real claim and not a straw one.
    """
    best, best_transformer = sel.SelectionReport.load().headline_pair()
    assert best.model_name == "pytorch_mlp_gpu"
    assert best_transformer.model_name == "distilbert"
    assert best.f1_macro > best_transformer.f1_macro
    assert best.f1_macro - best_transformer.f1_macro == pytest.approx(0.0704, abs=5e-5)


def test_the_headline_pair_was_never_scored_on_one_evaluation_set() -> None:
    """And second, that the two numbers were never comparable.

    One run's accuracy is a whole number of correct predictions over a multiple
    of 3,200; the other's over a multiple of 1,889, which is prime. A single
    evaluation set behind both needs 6,044,800 samples — fourteen times the
    entire training set this study builds.
    """
    best, best_transformer = sel.SelectionReport.load().headline_pair()
    divisors = [best.divisor, best_transformer.divisor]
    assert divisors == [3200, 1889]

    shared = sel.shared_evaluation_size([3200, 1889])
    assert shared == 6_044_800
    published = ds.DatasetReport.load().published["rows"]
    assert shared / published == pytest.approx(14.1, abs=0.05)


def test_the_gap_between_them_is_the_data_and_not_noise() -> None:
    """The honest version of the refutation, which is stronger than "it is noise".

    The two intervals do not overlap, so sampling error does not explain the
    difference. Something real separates the runs — and since they share no
    evaluation set, what separates them is what they were scored on.
    """
    best, best_transformer = sel.SelectionReport.load().headline_pair()
    low, high = best.interval, best_transformer.interval
    assert low is not None and high is not None
    assert low[0] > high[1]


# --- the submitted log, where the models at least share a corpus -------------


def test_the_ranking_below_the_top_two_depends_on_the_average() -> None:
    report = sel.SelectionReport.load()
    assert report.ranked(macro=False)[0].model_name == "distilbert"
    assert report.ranked(macro=True)[0].model_name == "distilbert"

    swings = {swing.label: swing for swing in report.swings()}
    assert swings["gru #6"].weighted_rank == 6
    assert swings["gru #6"].macro_rank == 3
    assert swings["gru #6"].places == 3
    # And the estimator that looks fifth-best on the reported column is
    # second-worst on the one recorded beside it.
    assert swings["linear_svc #4"].weighted_rank == 5
    assert swings["linear_svc #4"].macro_rank == 7
    # The top two never move, which is why the chapter's conclusion is about
    # the rest of the table.
    assert "distilbert #7" not in swings and "distilroberta #8" not in swings


def test_the_reported_columns_are_weighted_averages() -> None:
    """Recall weighted by support is accuracy. It is what identifies the column.

    The spreadsheet labels its columns plainly "Precision", "Recall" and "F1
    score" and never says which average. The identity holds for weighted and
    not for macro, so it settles the question — and the whole averaging finding
    rests on knowing the answer.
    """
    report = sel.SelectionReport.load()
    for row in report.submitted:
        assert row.recall_weighted == pytest.approx(row.accuracy, abs=5e-5)
        assert row.f1_macro < row.f1_weighted


def test_the_submitted_rows_are_not_one_dataset_either() -> None:
    """The four classical rows capped Neutral at 10,000 and the neural rows at 6,000.

    Neutral is the class the model is worst at, so how much of it a run had to
    classify changes the score. The transformers were given the smaller share,
    and the log's own preprocessing column says so.
    """
    caps = sel.SelectionReport.load().neutral_caps()
    assert sorted(caps) == [6000, 10000]
    assert {row.model_name for row in caps[10000]} == {
        "logistic_regression",
        "naive_bayes",
        "linear_svc",
    }
    assert {row.model_name for row in caps[6000]} == {
        "lstm",
        "gru",
        "distilbert",
        "distilroberta",
    }


# --- the two records do not meet ---------------------------------------------


def test_no_submitted_score_appears_in_the_run_log() -> None:
    """So the figures that were handed in cannot be traced to a recorded run.

    Every submitted macro F1 lands near a logged one — within 0.0002 to 0.0032 —
    and none of them lands on it. Near is what you would expect of the same
    configuration run again, and it is not the same as recorded.
    """
    nearest = sel.SelectionReport.load().nearest_logged()
    assert len(nearest) == 8
    assert all(match.gap > 0 for match in nearest)
    gaps = [match.gap for match in nearest]
    assert min(gaps) == pytest.approx(0.000206, abs=5e-7)
    assert max(gaps) == pytest.approx(0.003228, abs=5e-7)


def test_matching_by_value_identifies_nothing() -> None:
    """Three of the eight nearest scores were not even set by the same kind of model."""
    nearest = sel.SelectionReport.load().nearest_logged()
    off_type = [match for match in nearest if not match.same_model_type]
    assert len(off_type) == 3
    # The submitted LSTM's closest logged score belongs to a logistic regression.
    lstm = next(match for match in nearest if match.submitted.model_name == "lstm")
    assert sel.model_type_of(lstm.run.model_type) == "traditional_ml"


def test_model_type_normalises_the_spelling_and_nothing_else() -> None:
    """The two logs spell the same three categories differently."""
    assert sel.model_type_of("Traditional ML") == "traditional_ml"
    assert sel.model_type_of("traditional_ml") == "traditional_ml"
    assert sel.model_type_of("Deep Learning - RNN") == "deep_learning"
    assert sel.model_type_of("Transformer") == "transformer"


def test_a_source_link_is_kept_out_of_a_runs_summary() -> None:
    report = sel.SelectionReport.load()
    linked = next(run for run in report.runs if "http" in run.notes)
    assert "http" not in linked.summary
    assert not linked.summary.endswith(("-", "|", " "))
    plain = next(run for run in report.runs if "http" not in run.notes)
    assert plain.summary == plain.notes


# --- what the consistency checks catch ---------------------------------------


def _mutated(tmp_path: Path, mutate: object) -> sel.SelectionReport:
    raw = json.loads(Path(sel.DEFAULT_SUBMITTED_LOG).read_text(encoding="utf-8"))
    mutate(raw)  # type: ignore[operator]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")
    return sel.SelectionReport.load(sel.DEFAULT_RUN_LOG, path)


def test_a_recall_that_is_not_the_accuracy_is_caught(tmp_path: Path) -> None:
    """Because the whole averaging finding depends on that identity holding."""

    def break_recall(raw: dict[str, object]) -> None:
        rows = raw["rows"]
        assert isinstance(rows, list)
        rows[0]["recall_weighted"] = 0.5

    problems = sel.check_consistency(_mutated(tmp_path, break_recall))
    assert any("not weighted averages" in p for p in problems)


def test_a_macro_score_above_its_weighted_one_is_caught(tmp_path: Path) -> None:
    def break_averages(raw: dict[str, object]) -> None:
        rows = raw["rows"]
        assert isinstance(rows, list)
        rows[2]["f1_macro"] = 0.99

    problems = sel.check_consistency(_mutated(tmp_path, break_averages))
    assert any("rare classes beat the common ones" in p for p in problems)


def test_a_neutral_cap_that_contradicts_its_own_preprocessing_is_caught(
    tmp_path: Path,
) -> None:
    """The cap is read out of the preprocessing text, so the two cannot drift."""

    def break_cap(raw: dict[str, object]) -> None:
        rows = raw["rows"]
        assert isinstance(rows, list)
        rows[0]["neutral_cap"] = 4000

    problems = sel.check_consistency(_mutated(tmp_path, break_cap))
    assert any("is not what" in p for p in problems)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("accuracy", 1.4, "not a rate"),
        ("training_time_s", 0.0, "training time is"),
        ("nr", 9, "not 1..8 in order"),
    ],
)
def test_a_submitted_row_that_cannot_be_right_is_caught(
    tmp_path: Path, field: str, value: object, expected: str
) -> None:
    def break_field(raw: dict[str, object]) -> None:
        rows = raw["rows"]
        assert isinstance(rows, list)
        rows[0][field] = value

    problems = sel.check_consistency(_mutated(tmp_path, break_field))
    assert any(expected in p for p in problems)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("f1_macro", "1.2", "not a rate"),
        ("training_time", "0", "training time is"),
        ("iteration_id", "77", "not 1..101 in order"),
    ],
)
def test_a_logged_run_that_cannot_be_right_is_caught(
    tmp_path: Path, column: str, value: str, expected: str
) -> None:
    lines = Path(sel.DEFAULT_RUN_LOG).read_text(encoding="utf-8").splitlines()
    fields = lines[0].split(",")
    row = lines[1].split(",")
    row[fields.index(column)] = value
    lines[1] = ",".join(row)
    path = tmp_path / "broken.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    report = sel.SelectionReport.load(path, sel.DEFAULT_SUBMITTED_LOG)
    assert any(expected in p for p in sel.check_consistency(report))


# --- figures -----------------------------------------------------------------


def test_the_figures_are_stamped_with_both_logs_together(tmp_path: Path) -> None:
    """One digest over the pair: touching either log makes both figures stale."""
    from emotion_timeline import figures as shared

    report = sel.SelectionReport.load()
    for path in sel_figures.render_all(report, tmp_path):
        assert shared.read_stamp(path) == report.digest
    assert sel_figures.check_figures_current(report, tmp_path) == []


def test_changing_either_log_makes_the_figures_stale(tmp_path: Path) -> None:
    report = sel.SelectionReport.load()
    sel_figures.render_all(report, tmp_path)

    def nudge(raw: dict[str, object]) -> None:
        rows = raw["rows"]
        assert isinstance(rows, list)
        rows[0]["training_time_s"] = 220.0

    stale = sel_figures.check_figures_current(_mutated(tmp_path, nudge), tmp_path)
    assert len(stale) == len(sel_figures.FIGURES)
    assert all("drawn from" in problem for problem in stale)


def test_the_committed_model_selection_figures_are_current() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"
    assert sel_figures.check_figures_current(sel.SelectionReport.load(), assets) == []


def _with_a_rounded_best_run(tmp_path: Path) -> sel.SelectionReport:
    """The committed log with the best run's accuracy rounded to four places."""
    lines = Path(sel.DEFAULT_RUN_LOG).read_text(encoding="utf-8").splitlines()
    fields = lines[0].split(",")
    row = lines[70].split(",")  # iteration 70, pytorch_mlp_gpu, the best run
    assert row[fields.index("model_name")] == "pytorch_mlp_gpu"
    row[fields.index("accuracy")] = "0.8653"
    lines[70] = ",".join(row)
    path = tmp_path / "rounded.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return sel.SelectionReport.load(path, sel.DEFAULT_SUBMITTED_LOG)


def test_a_rounded_headline_accuracy_is_reported_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """Rounding is not an inconsistency, so the gate lets it through.

    That means the description has to cope with a headline run whose evaluation
    set cannot be recovered, and say so, rather than assume there is a divisor
    to print.
    """
    report = _with_a_rounded_best_run(tmp_path)
    assert sel.check_consistency(report) == []
    text = "\n".join(sel.describe(report))
    assert "logged rounded" in text
    assert "unsupported" in text
    assert "6,044,800" not in text


def test_a_rounded_headline_accuracy_leaves_it_off_the_figure(tmp_path: Path) -> None:
    """The picture drops what it cannot place rather than inventing a position."""
    report = _with_a_rounded_best_run(tmp_path)
    assert sel_figures.render_all(report, tmp_path)
    assert sel_figures.check_figures_current(report, tmp_path) == []


def test_the_describe_lines_say_what_the_docs_say() -> None:
    lines = sel.describe(sel.SelectionReport.load())
    text = "\n".join(lines)
    assert "8 submitted rows, 101 logged runs" in text
    assert "27 distinct divisors" in text
    assert "6,044,800 samples" in text
    assert "3 places up" in text
    assert "1 place down" in text  # singular, because it reads as prose


def test_the_early_runs_all_sit_on_the_client_sized_evaluation_set() -> None:
    """The provenance claim in the docs, pinned because it is a published claim.

    The notes column first mentions external data at iteration 48. Every
    placeable run before that recovers a divisor of either 112 or 93 -- a set of
    roughly a hundred rows across seven classes, which is why so many of those
    runs collapsed onto a single answer. Afterwards the divisors climb into the
    thousands and only one of them still divides either number.

    The wording of this was wrong on the first pass: the docs said the early
    divisors were "112, 93, 56 or 31", which misses the runs that recover 7 and
    3. Those divide 112 and 93 respectively, so the finding held, but the list
    did not. Stating it as divisibility is both accurate and stronger.
    """
    report = sel.SelectionReport.load()
    early = [(run, d) for run, d in sel.placed(report.runs) if run.iteration < 48]
    late = [(run, d) for run, d in sel.placed(report.runs) if run.iteration >= 48]

    assert early and late
    assert all(112 % d == 0 or 93 % d == 0 for _, d in early)
    assert {d for _, d in early} == {3, 7, 31, 56, 93, 112}
    assert {d for _, d in late if 112 % d == 0 or 93 % d == 0} == {28}
    assert max(d for _, d in late) == 8682
