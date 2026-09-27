from homework_1_2.tema2 import search_keyword


def test_search_keyword_finds_pytest_file():
    result = search_keyword()

    assert "test_run_agent.py" in result
