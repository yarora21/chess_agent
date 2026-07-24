"""Phase 5: the faithfulness harness.

These tests matter more than the narration itself. The harness is the only thing
standing between a fluent language model and a confidently invented statistic,
and a check that never fires is worse than no check -- so most of these feed it
bad text and assert it complains.
"""

from __future__ import annotations

from chess_agent import narrate

RECORDS = [
    narrate.Record("m0", "blunder rate: first 6.2%, last 4.1%, 23080 moves",
                   [6.2, 4.1, 23080, 100]),
    narrate.Record("m1", "book depth: 3.5 to 4.9 plies", [3.5, 4.9]),
]


# ---------------------------------------------------------------------------
# Numbers must trace to a record
# ---------------------------------------------------------------------------

def test_text_using_only_record_numbers_passes():
    text = "Your blunder rate went from 6.2% to 4.1% across 23080 moves."
    report = narrate.check(text, RECORDS)
    assert report.ok, report.render()
    assert report.numbers_checked == 3


def test_an_invented_number_is_caught():
    text = "Your blunder rate went from 6.2% to 4.1%, a drop of 33.9%."
    report = narrate.check(text, RECORDS)
    assert not report.ok
    assert any("33.9" in u for u in report.unsupported)


def test_a_plausible_but_unsupported_number_is_caught():
    """The dangerous case: a number that looks right and is not in the records."""
    text = "Your blunder rate improved to 4.0%."
    report = narrate.check(text, RECORDS)
    assert not report.ok, "4.0 is close to 4.1 but is not a record value"


def test_arithmetic_the_model_did_itself_is_caught():
    text = "You blundered 1421 times in total."
    report = narrate.check(text, RECORDS)
    assert not report.ok, "the model must never compute a new number"


def test_unsupported_numbers_report_their_context():
    report = narrate.check("Your rate fell to 99.9% last month.", RECORDS)
    assert report.unsupported and "99.9" in report.unsupported[0]
    assert "rate fell to" in report.unsupported[0]


def test_small_bare_integers_are_allowed_as_ordinary_english():
    report = narrate.check("There are 2 things worth noting.", RECORDS)
    assert report.ok, "small integers are prose, not claims"


def test_integer_form_of_a_record_value_is_accepted():
    report = narrate.check("Across 23080 moves.", RECORDS)
    assert report.ok


def test_thousands_separators_are_understood():
    report = narrate.check("Across 23,080 moves.", RECORDS)
    assert report.ok, "23,080 is the same number as 23080"


def test_a_number_at_the_end_of_a_sentence_is_parsed_without_the_period():
    report = narrate.check("The rate was 6.2.", RECORDS)
    assert report.ok


# ---------------------------------------------------------------------------
# Verdict language is forbidden -- the MVP has no significance testing
# ---------------------------------------------------------------------------

def test_claiming_significance_is_rejected():
    text = "Your blunder rate dropped significantly, from 6.2% to 4.1%."
    report = narrate.check(text, RECORDS)
    assert not report.ok
    assert "significantly" in report.banned_phrases


def test_statistical_language_is_rejected():
    for phrase in ("a statistically sound trend", "this proves you improved",
                   "the confidence interval excludes zero"):
        report = narrate.check(phrase, RECORDS)
        assert not report.ok, f"{phrase!r} should be rejected"


def test_descriptive_phrasing_is_accepted():
    text = "Your blunder rate went from 6.2% to 4.1%. That is a description of "
    text += "what the numbers did, not a claim about why."
    assert narrate.check(text, RECORDS).ok


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def test_records_carry_the_values_that_appear_in_their_text():
    record = narrate.Record("x", "rate was 12.5% over 400 moves", [12.5, 400])
    assert narrate.check("12.5% of 400", [record]).ok


def test_records_render_with_a_citable_id():
    assert RECORDS[0].render().startswith("[m0] ")


def test_the_system_prompt_forbids_chess_judgment_and_verdicts():
    prompt = narrate.SYSTEM_PROMPT.lower()
    assert "never make a chess judgment" in prompt
    assert "descriptive only" in prompt
    assert "sample size" in prompt


def test_a_hyphen_joining_words_is_not_a_minus_sign():
    """"Maia-1200" is a model name, not negative 1200 -- this was a real false positive."""
    records = [narrate.Record("r", "maia 1200 rate 43.7%", [1200, 43.7])]
    report = narrate.check("You played the Maia-1200 move 43.7% of the time.", records)
    assert report.ok, report.render()


def test_a_genuine_negative_number_is_still_read_as_negative():
    records = [narrate.Record("r", "slope was -4.5", [-4.5])]
    assert narrate.check("The slope was -4.5 per game.", records).ok
    assert not narrate.check("The slope was 4.5 per game.", records).ok


def test_a_trailing_period_is_not_read_as_a_decimal_point():
    records = [narrate.Record("r", "rate 7.9 over 100", [7.9, 100])]
    assert narrate.check("The rate was 7.9. That is the figure.", records).ok
