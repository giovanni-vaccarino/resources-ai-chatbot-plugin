"""Unit Tests for sparse_retrieve module (keyword-based sparse retriever)."""

from rag.retriever.retriever_bm25 import perform_keyword_search, search_bm25_index


def test_perform_keyword_search_empty_query(mocker):
    """Empty query should log a warning and return an empty list."""
    mock_logger = mocker.Mock()

    results = perform_keyword_search(
        query="   ",
        logger=mock_logger,
        source_name="plugins",
        keyword_threshold=0.5,
        top_k=5,
    )

    mock_logger.warning.assert_called_once_with("Empty query received.")
    assert results == []


def test_perform_keyword_search_no_index(mocker):
    """Missing index should return an empty list."""
    mock_logger = mocker.Mock()

    mocker.patch("rag.retriever.retriever_bm25.indexer", return_value=None)

    mocker.patch(
        "rag.retriever.retriever_bm25.load_vector_index",
        return_value=("unused_index", [{"id": "doc1"}]),
    )

    results = perform_keyword_search(
        query="valid query",
        logger=mock_logger,
        source_name="plugins",
        keyword_threshold=0.5,
        top_k=5,
    )

    assert results == []


def test_perform_keyword_search_no_metadata(mocker):
    """Missing metadata should return an empty list."""
    mock_logger = mocker.Mock()

    mock_index = mocker.Mock()
    mocker.patch("rag.retriever.retriever_bm25.indexer", return_value=mock_index)

    mocker.patch(
        "rag.retriever.retriever_bm25.load_vector_index",
        return_value=("unused_index", None),
    )

    results = perform_keyword_search(
        query="valid query",
        logger=mock_logger,
        source_name="plugins",
        keyword_threshold=0.5,
        top_k=5,
    )

    assert results == []


def test_perform_keyword_search_success_with_threshold_filter(mocker):
    """Successful search with threshold filtering, ensuring call args and output shape."""
    mock_logger = mocker.Mock()

    mock_index = mocker.Mock()
    mock_get = mocker.patch("rag.retriever.retriever_bm25.indexer.get", return_value=mock_index)

    mock_metadata = [{"id": "1"}, {"id": "2"}]
    mock_load = mocker.patch(
        "rag.retriever.retriever_bm25.load_vector_index",
        return_value=("unused_index", mock_metadata),
    )

    mock_search = mocker.patch(
        "rag.retriever.retriever_bm25.search_bm25_index",
        return_value=([{"id": "1"}, {"id": "2"}], [0.8, 0.2]),
    )

    results = perform_keyword_search(
        query="valid query",
        logger=mock_logger,
        source_name="plugins",
        keyword_threshold=0.5,
        top_k=3,
    )

    mock_get.assert_called_once_with("plugins")
    mock_load.assert_called_once_with(mock_logger, "plugins")
    mock_search.assert_called_once_with("valid query", mock_index, mock_metadata, mock_logger, 3)

    assert results == [{"chunk": {"id": "1"}, "score": 0.8}]


def test_perform_keyword_search_success_all_filtered_out(mocker):
    """All results are below threshold; returns empty list."""
    mock_logger = mocker.Mock()

    mock_index = mocker.Mock()
    mocker.patch("rag.retriever.retriever_bm25.indexer", return_value=mock_index)

    mock_metadata = [{"id": "1"}]
    mocker.patch(
        "rag.retriever.retriever_bm25.load_vector_index",
        return_value=("unused_index", mock_metadata),
    )

    mocker.patch(
        "rag.retriever.retriever_bm25.search_bm25_index",
        return_value=([{"id": "1"}], [0.3]),
    )

    results = perform_keyword_search(
        query="valid query",
        logger=mock_logger,
        source_name="plugins",
        keyword_threshold=0.9,
        top_k=1,
    )

    assert results == []


def test_search_bm25_index_success_with_missing_metadata_logs_warning(mocker):
    """search_bm25_index should map ids to metadata, warn for missing, and return matched results & scores."""
    mock_logger = mocker.Mock()

    metadata = [{"id": "a", "content": "A content"}]

    class DummyIndex:
        def search(self, query, return_docs, cutoff):
            assert query == "q"
            assert return_docs is True
            assert cutoff == 2
            return [
                {"id": "a", "score": 0.7},
                {"id": "b", "score": 0.4},
            ]

    index = DummyIndex()

    data, scores = search_bm25_index(
        query="q",
        index=index,
        metadata=metadata,
        logger=mock_logger,
        top_k=2,
    )

    assert data == [{"id": "a", "content": "A content"}]
    assert scores == [0.7, 0.4]

    mock_logger.warning.assert_called_once_with("No metadata found for chunk ID: %s", "b")


def test_search_bm25_index_empty_results(mocker):
    """When index.search returns no results, both data and scores should be empty."""
    mock_logger = mocker.Mock()

    class EmptyIndex:
        def search(self, query, return_docs, cutoff):
            return []

    data, scores = search_bm25_index(
        query="anything",
        index=EmptyIndex(),
        metadata=[],
        logger=mock_logger,
        top_k=10,
    )

    assert data == []
    assert scores == []
