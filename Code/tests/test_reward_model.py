"""Tests for reward calculator."""

from mathlm.data import GSM8KExample
from mathlm.rewards import RewardCalculator, RewardWeights


def _example():
    return GSM8KExample(uid="1", question="1+1?", answer="2")


def test_reward_calculator_awards_components():
    calc = RewardCalculator(weights=RewardWeights())
    output = """
Let's reason it out.
We add two numbers step by step to be clear for the grader and make sure the reasoning has enough tokens.
```python
result = 1 + 1
print(result)
```
Therefore the answer is 2.
"""
    breakdown = calc.evaluate(_example(), output)
    assert breakdown.syntax_reward > 0
    assert breakdown.execution_reward > 0
    assert breakdown.extraction_reward > 0
    assert breakdown.reasoning_reward > 0
    assert breakdown.exact_reward > 0
    assert breakdown.total > 0


def test_reward_calculator_penalizes_incorrect_answer():
    calc = RewardCalculator(weights=RewardWeights())
    output = "Final answer is 5"
    breakdown = calc.evaluate(_example(), output)
    assert breakdown.exact_reward == 0
    assert breakdown.penalty < 0
