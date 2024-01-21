import pytest

from app import _1_pull_transcripts


@pytest.mark.skip(reason="Not self-contained, might break local database")
def test_pull_transcripts():
    print("starting test")
    _1_pull_transcripts.main()
    print("test complete")
