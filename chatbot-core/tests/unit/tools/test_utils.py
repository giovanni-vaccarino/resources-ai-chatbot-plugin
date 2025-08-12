from api.tools.utils import get_default_tools_call, validate_tool_calls, retrieve_documents, extract_top_chunks
from types import MappingProxyType
import api.tools.utils as utils

def test_get_default_tools_call_basic():
    query = "jenkins pipeline syntax"
    expected = [
        {"tool": "search_jenkins_docs", "params": {"query": query}},
        {"tool": "search_plugin_docs", "params": {"plugin_name": None, "query": query}},
        {"tool": "search_stackoverflow_threads", "params": {"query": query}},
        {"tool": "search_community_threads", "params": {"query": query}},
    ]

    result = get_default_tools_call(query)

    assert isinstance(result, list)
    assert len(result) == 4
    assert result == expected

def test_validate_tool_calls_all_valid(mocker):
    logger = mocker.Mock()

    calls = [
        {"tool": "search_plugin_docs", "params": {"plugin_name": "git", "query": "foo"}},
        {"tool": "search_jenkins_docs", "params": {"query": "bar"}},
        {"tool": "search_stackoverflow_threads", "params": {"query": "baz"}},
        {"tool": "search_community_threads", "params": {"query": "qux"}},
    ]

    ok = validate_tool_calls(calls, logger)

    assert ok is True
    logger.warning.assert_not_called()


def test_validate_tool_calls_invalid_tool_name(mocker):
    logger = mocker.Mock()
    calls = [{"tool": "not_a_tool", "params": {}}]

    ok = validate_tool_calls(calls, logger)

    assert ok is False
    logger.warning.assert_called_once_with("Tool %s not available.", "not_a_tool")


def test_validate_tool_calls_params_not_dict_using_patched_signatures(mocker):
    """
    Exercise the 'params is not a dict' branch safely by patching TOOL_SIGNATURES
    to an empty signature for a toy tool so we don't iterate/lookup keys on params.
    """
    logger = mocker.Mock()
    mocker.patch.object(
        utils,
        "TOOL_SIGNATURES",
        new=MappingProxyType({"toy_tool": {}}),
    )

    calls = [{"tool": "toy_tool", "params": "oops"}]  # not a dict

    ok = validate_tool_calls(calls, logger)

    assert ok is False
    logger.warning.assert_called_once_with("Params for tool %s is not a dict.", "toy_tool")


def test_validate_tool_calls_wrong_type_using_patched_signatures(mocker):
    """
    Validate the type-mismatch branch by patching a simple signature and
    passing a wrong-typed value.
    """
    logger = mocker.Mock()
    mocker.patch.object(
        utils,
        "TOOL_SIGNATURES",
        new=MappingProxyType({"toy_tool": {"query": str}}),
    )

    calls = [{"tool": "toy_tool", "params": {"query": 123}}]  # wrong type

    ok = validate_tool_calls(calls, logger)

    assert ok is False
    logger.warning.assert_called_with(
        "Tool: %s: Param %s is not of the expected type %s.",
        "toy_tool",
        "query",
        "str",
    )


def test_validate_tool_calls_mixed_validity_returns_false(mocker):
    """
    One valid and one invalid call -> overall result should be False.
    """
    logger = mocker.Mock()
    mocker.patch.object(
        utils,
        "TOOL_SIGNATURES",
        new=MappingProxyType({
            "search_jenkins_docs": {"query": str},
            "toy_tool": {"query": str},
        }),
    )

    calls = [
        {"tool": "search_jenkins_docs", "params": {"query": "ok"}},  # valid
        {"tool": "toy_tool", "params": {"query": 42}},               # invalid type
    ]

    ok = validate_tool_calls(calls, logger)
    assert ok is False


def test_retrieve_documents_calls_and_returns(mocker):
    # Arrange
    logger = mocker.Mock()
    model = mocker.Mock()

    # Patch retrieval_config used inside the function
    mocker.patch.object(
        utils,
        "retrieval_config",
        new={
            "top_k_semantic": 2,
            "keyword_threshold": 0.7,
            "top_k_keyword": 3,
            "empty_context_message": "[NO CONTEXT]",
        },
    )

    # Patch the two called functions
    mock_get = mocker.patch.object(
        utils,
        "get_relevant_documents",
        return_value=([{"id": "s1"}], [0.11]),
    )
    mock_kw = mocker.patch.object(
        utils,
        "perform_keyword_search",
        return_value=[
            {"chunk": {"id": "k1"}, "score": 0.9},
            {"chunk": {"id": "k2"}, "score": 0.8},
        ],
    )

    # Act
    data_sem, scores_sem, data_kw, scores_kw = retrieve_documents(
        query="q-text",
        keywords="k-text",
        logger=logger,
        source_name="plugins",
        embedding_model=model,
    )

    # Assert: calls carry through the config values
    mock_get.assert_called_once_with(
        "q-text",
        model,
        logger=logger,
        source_name="plugins",
        top_k=2,
    )
    mock_kw.assert_called_once_with(
        "k-text",
        logger,
        source_name="plugins",
        keyword_threshold=0.7,
        top_k=3,
    )

    # Assert: returns are correctly split/flattened
    assert data_sem == [{"id": "s1"}]
    assert scores_sem == [0.11]
    assert data_kw == [{"id": "k1"}, {"id": "k2"}]
    assert scores_kw == [0.9, 0.8]


def test_extract_top_chunks_happy_path_orders_by_heap_and_returns_content(mocker):
    # Arrange
    logger = mocker.Mock()

    sem = [{"id": "a", "chunk_text": "A"}]
    kw = [{"id": "b", "chunk_text": "B"}]

    # The function pops the smallest value first; more negative first.
    mock_get_inv = mocker.patch.object(
        utils,
        "get_inverted_scores",
        return_value=[(-0.9, "b"), (-0.8, "a")],
    )
    mock_extract = mocker.patch.object(
        utils,
        "extract_chunks_content",
        return_value="JOINED",
    )

    # Act
    out = extract_top_chunks(
        data_retrieved_semantic=sem,
        scores_semantic=[0.1],
        data_retrieved_keyword=kw,
        scores_keyword=[0.9],
        top_k=5,  # larger than available; should just take both
        logger=logger,
        semantic_weight=0.3,
    )

    # Assert: get_inverted_scores called with ids and provided scores
    mock_get_inv.assert_called_once_with(
        ["a"], [0.1], ["b"], [0.9], 0.3
    )
    # Assert: chunks passed to extract follow heap order: 'b' then 'a'
    mock_extract.assert_called_once_with([kw[0], sem[0]], logger)
    assert out == "JOINED"


def test_extract_top_chunks_empty_scores(mocker):

    logger = mocker.Mock()

    mocker.patch.object(utils, "get_inverted_scores", return_value=[])
    mock_extract = mocker.patch.object(utils, "extract_chunks_content", return_value="[EMPTY]")

    out = extract_top_chunks(
        data_retrieved_semantic=[],
        scores_semantic=[],
        data_retrieved_keyword=[],
        scores_keyword=[],
        top_k=3,
        logger=logger,
        semantic_weight=0.5,
    )

    mock_extract.assert_called_once_with([], logger)
    assert out == "[EMPTY]"
