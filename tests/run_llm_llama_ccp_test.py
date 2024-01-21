import pytest

from app import _3_run_llm_llama_cpp


@pytest.mark.skip(reason="Not self-contained, require OpenAI key")
def test_run_llama():
    print("starting test")
    _3_run_llm_llama_cpp.main()
    print("test complete")
