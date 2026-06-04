"""embedding-worker — search + embeddings (master plan Phase 8 / spec/11 A4.3).

Indexes platform artifacts (events, vlm_observations) as pgvector embeddings and
answers natural-language video search via a parse -> metadata-filter ->
vector-search -> fuse -> critic pipeline. The embedding model (C-RADIOv4-H) and
the critic (Nemotron) are both optional: a deterministic fake_infer / stub path
keeps the worker, the A11 plugin test, and CI runnable with no GPU / weights / DB.
"""
from __future__ import annotations

# Tolerate both package use and the repo's path-based test setup (the hyphenated
# dir name `embedding-worker` isn't a valid module name).
try:
    from .plugin import EMBED_DIM, NvidiaEmbeddingPlugin, default_config
    from .indexer import (event_text_repr, index_all, index_events,
                          index_vlm_observations, upsert_embedding, vlm_text_repr)
    from .search import (Candidate, ParsedQuery, SearchCriticStub, fuse,
                        metadata_filter, query_parser, search, vector_search)
except ImportError:                               # path-based / direct import
    from plugin import EMBED_DIM, NvidiaEmbeddingPlugin, default_config
    from indexer import (event_text_repr, index_all, index_events,
                         index_vlm_observations, upsert_embedding, vlm_text_repr)
    from search import (Candidate, ParsedQuery, SearchCriticStub, fuse,
                       metadata_filter, query_parser, search, vector_search)

__all__ = [
    "EMBED_DIM", "NvidiaEmbeddingPlugin", "default_config",
    "event_text_repr", "vlm_text_repr", "upsert_embedding",
    "index_events", "index_vlm_observations", "index_all",
    "ParsedQuery", "Candidate", "SearchCriticStub",
    "query_parser", "metadata_filter", "vector_search", "fuse", "search",
]
