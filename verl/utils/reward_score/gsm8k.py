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

import re
from decimal import Decimal, InvalidOperation

_SOLUTION_CLIP_CHARS = 300

# Everything after the '####' marker on that line. Deliberately looser than the
# old "#### (\-?[0-9\.\,]+)", which required a digit IMMEDIATELY after the
# space: DeepSeek-V4 writes "#### $36", the '$' blocked the match, and the
# .replace("$", "") that was plainly meant to handle it ran only on the regex
# result -- so it never got the chance. On a 128-sample GSM8K dump, 12 of the 18
# zero-score generations were correct answers rejected this way (67% of all
# zeros). The marker is still required, so strict mode still tests the format;
# what changed is that a currency symbol or a trailing word no longer counts as
# a missing answer.
_STRICT_MARKER = re.compile(r"####\s*([^\n]*)")
# First number on that line: optional sign, optional thousands separators, optional
# fraction. Anchored to a digit so "$" / "-" alone cannot match.
_LEADING_NUMBER = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _normalize_number(text):
    """Strip the decorations that are not part of the value ('$', '1,234', '36.')."""
    return text.replace(",", "").replace("$", "").strip().rstrip(".")


def answers_match(answer, ground_truth):
    """Numeric comparison, falling back to the original exact-string rule.

    '10.00' and '10' are the same answer; string equality scored the first as
    wrong. Decimal rather than float so the comparison is exact and 0.1+0.2 style
    representation error cannot decide a reward. Non-numeric ground truths fall
    back to string equality, so datasets that carry them behave as before.
    """
    a, b = _normalize_number(str(answer)), _normalize_number(str(ground_truth))
    try:
        return Decimal(a) == Decimal(b)
    except (InvalidOperation, ValueError):
        return a == b


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model: the '####' marker is still
        # required. Only the number's own decoration is tolerated (see the
        # patterns above for why).
        marked = _STRICT_MARKER.findall(solution_str)
        final_answer = None
        for tail in reversed(marked):  # take the last marker that carries a number
            number = _LEADING_NUMBER.search(tail)
            if number is not None:
                final_answer = _normalize_number(number.group(0))
                break
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answers_match(answer, ground_truth):
            return score
        else:
            return format_score
