import pytest

from app import query_vector_db


@pytest.mark.skip(reason="Expects user-input")
def test_query_vector_db():
    print("starting test")
    query_vector_db.main()
    print("test complete")
