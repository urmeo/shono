import pytest

from shono.eval import cer, score, wer


@pytest.mark.parametrize("compute", [wer, cer, score])
@pytest.mark.parametrize(
    "references,hypotheses",
    [
        ("আমি", "আমি"),
        (["আমি"], [None]),
        ([7], ["আমি"]),
        (["আমি"], [False]),
        ([b"word"], ["আমি"]),
    ],
)
def test_invalid_text_inputs_fail_clearly(compute, references, hypotheses):
    with pytest.raises(ValueError, match="strings"):
        compute(references, hypotheses)
