"""verifier 单测（M6 验收项）：正常提取、格式变体、提取失败 case。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.verifier import exact_match, extract_answer, score_responses


class TestExtractAnswer:
    def test_plain_integer(self):
        assert extract_answer("Step by step...\n#### 72") == 72.0

    def test_last_marker_wins(self):
        # 多次出现 #### 时取最后一个（防中间步骤误标）
        assert extract_answer("#### 10 is wrong\nrecompute\n#### 42") == 42.0

    def test_comma_thousands(self):
        assert extract_answer("#### 1,000") == 1000.0

    def test_dollar_sign(self):
        assert extract_answer("#### $18") == 18.0

    def test_decimal(self):
        assert extract_answer("#### 3.5") == 3.5

    def test_negative(self):
        assert extract_answer("#### -12") == -12.0

    def test_fraction(self):
        assert extract_answer("#### 1/2") == 0.5

    def test_missing_marker_returns_none(self):
        assert extract_answer("The answer is 72.") is None

    def test_marker_without_number_returns_none(self):
        assert extract_answer("final answer:\n####") is None

    def test_empty_text(self):
        assert extract_answer("") is None
        assert extract_answer(None) is None

    def test_division_by_zero_fraction(self):
        assert extract_answer("#### 1/0") is None


class TestExactMatch:
    def test_match_against_gold_text(self):
        assert exact_match("...\n#### 72", "Janet makes $18 - $3 = ... #### 72")

    def test_mismatch(self):
        assert not exact_match("#### 71", "#### 72")

    def test_extract_failure_is_wrong(self):
        assert not exact_match("no marker here", "#### 72")


class TestScoreResponses:
    def test_rewards_and_fail_rate(self):
        rewards, fail = score_responses(
            ["#### 5", "#### 6", "oops no marker"],
            ["#### 5", "#### 7", "#### 8"],
        )
        assert rewards == [1.0, 0.0, 0.0]
        assert fail == pytest.approx(1 / 3)

    def test_empty_input(self):
        assert score_responses([], []) == ([], 0.0)
