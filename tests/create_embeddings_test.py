import pytest

from app import _2_create_embeddings


@pytest.mark.skip(reason="Not self-contained, require OpenAI key")
def test_create_embeddings():
    print("starting test")
    _2_create_embeddings.main()
    print("test complete")
