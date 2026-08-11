# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The GSM8K grader must not reject correct answers over their decoration.

On a 128-generation DeepSeek-V4 dump, 18 samples scored 0 and **12 of them were
correct** -- rejected because the model writes ``#### $36`` and the old pattern
required a digit immediately after the marker, while the ``.replace("$", "")``
meant to handle exactly that ran only on the regex result. Separately ``10.00``
compared as a string against ``10`` also failed.

That matters beyond the score: GRPO's advantage is the within-group reward
variance, so if the only variance is "did it print a dollar sign", the gradient
is training formatting rather than reasoning.

The strict method still REQUIRES the ``####`` marker -- that part is the format
test and is intentionally preserved.
"""

import pytest

from verl.utils.reward_score.gsm8k import answers_match, compute_score, extract_solution


@pytest.mark.parametrize(
    "solution,ground_truth,reason",
    [
        ("#### $36", "36", "currency symbol -- the case found in the DSv4 dump"),
        ("#### $10.00", "10", "currency symbol and a trailing zero fraction"),
        ("#### 10.00", "10", "10.00 is 10"),
        ("#### 10", "10.00", "and the comparison is symmetric"),
        ("#### 1,234", "1234", "thousands separator"),
        ("#### $10,000", "10000", "both at once, as the model actually writes money"),
        ("#### 36.", "36", "sentence-final period after the answer"),
        ("#### -5", "-5", "negative answer"),
        ("#### $1,234.50", "1234.5", "separator, symbol and fraction together"),
        ("#### 36 apples", "36", "unit word after the number"),
        ("...output after \"####\".</think>#### $15", "15", "marker inside a reasoning trace"),
        ("#### 12\n#### 36", "36", "last marker wins, as before"),
    ],
)
def test_correct_answers_are_accepted(solution, ground_truth, reason):
    assert compute_score(solution, ground_truth) == 1.0, reason


@pytest.mark.parametrize(
    "solution,ground_truth,reason",
    [
        ("the answer is 36", "36", "no marker at all -- strict mode still tests the format"),
        ("#### abc", "36", "marker with no number in it"),
        ("#### ", "36", "empty marker"),
    ],
)
def test_missing_answer_scores_zero(solution, ground_truth, reason):
    assert compute_score(solution, ground_truth) == 0, reason


def test_wrong_answer_gets_format_score_not_full():
    """A parsed-but-wrong answer must stay distinguishable from an unparsable one."""
    assert compute_score("#### 37", "36") == 0.0
    assert compute_score("#### 37", "36", format_score=0.1) == 0.1
    # ... and an unparsable one is not eligible for format_score
    assert compute_score("no marker here", "36", format_score=0.1) == 0


def test_extract_solution_strips_decoration():
    assert extract_solution("#### $1,234.50") == "1234.50"
    assert extract_solution("#### 36.") == "36"


def test_flexible_method_still_works():
    assert compute_score("the answer is 36", "36", method="flexible") == 1.0


def test_answers_match_falls_back_to_string_equality():
    """Non-numeric ground truths must behave exactly as the old comparison did."""
    assert answers_match("yes", "yes")
    assert not answers_match("yes", "no")


def test_answers_match_is_exact_not_float_approximate():
    """Decimal, not float: representation error must never decide a reward."""
    assert answers_match("0.1", "0.10")
    assert not answers_match("0.1", "0.11")


def test_long_solution_is_still_clipped_from_the_end():
    """The 300-char clip is a speed optimisation; the answer lives at the end."""
    assert compute_score("x" * 5000 + "\n#### $36", "36") == 1.0
    # a marker only in the discarded head is still invisible, as before
    assert compute_score("#### 36" + "y" * 5000, "36") == 0
