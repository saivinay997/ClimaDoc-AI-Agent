import os
from functools import lru_cache
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore


@lru_cache()
def _load_file_secrets() -> dict:
    """
    Load secrets from TOML files commonly used by Streamlit.
    Priority:
    1) .streamlit/secrets.toml
    2) secrets.toml
    """
    candidate_paths = [
        Path(".streamlit") / "secrets.toml",
        Path("secrets.toml"),
    ]
    for path in candidate_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return tomllib.load(f)
            except Exception:
                # If parsing fails, continue to next candidate or fallback to env
                pass
    return {}


@lru_cache()
def _load_streamlit_secrets() -> dict:
    """Attempt to load secrets from the Streamlit runtime if available."""
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:
        return {}


def get_secret(key: str, default=None):
    """
    Retrieve a secret value with the following precedence:
    1) Streamlit secrets (if running under Streamlit)
    2) Local TOML secrets (.streamlit/secrets.toml or secrets.toml)
    3) Environment variables
    """
    streamlit_secrets = _load_streamlit_secrets()
    if key in streamlit_secrets:
        return streamlit_secrets.get(key, default)

    file_secrets = _load_file_secrets()
    if key in file_secrets:
        return file_secrets.get(key, default)

    return os.getenv(key, default)

