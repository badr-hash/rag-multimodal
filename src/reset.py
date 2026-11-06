from __future__ import annotations


def reset_tables():
    """Clear the in-memory stores used by the app."""
    try:
        from store import reset_vector_stores
        reset_vector_stores()
    except Exception:
        pass
    try:
        import images
        images._CLIP_STORE_CACHE.clear()
    except Exception:
        pass
    return None
