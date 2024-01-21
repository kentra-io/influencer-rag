import pytest

from app import evaluate_all_questions


@pytest.mark.skip(reason="Not self-contained, require OpenAI key")
def test_evaluate_all_questions():
    print("starting test")
    evaluate_all_questions.main()
    print("test complete")
