# This local package provides event_tokenizer and remi_wrapper.
# It shares the name "tokenizers" with the HuggingFace tokenizers library,
# which miditok imports internally via `from tokenizers import AddedToken`.
# To prevent that import from failing, we re-export the HuggingFace library's
# public API into this namespace.
#
# We locate the real HuggingFace package with PathFinder against a search path
# that excludes the project root, load it under the canonical "tokenizers" name
# (so its own submodules register as tokenizers.* correctly), copy its public
# names into our namespace, then restore ourselves as the authoritative
# "tokenizers" module. This avoids the infinite recursion that a naive
# sys.path-removal approach hits when the project root is also the cwd or is
# served from a cached path finder.

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

_our_dir = Path(__file__).resolve().parent       # <project>/tokenizers
_our_parent = _our_dir.parent                     # <project>


def _resolves_to_parent(entry: str) -> bool:
    """True if a sys.path entry points at the project root (where THIS package lives)."""
    try:
        return Path(entry or os.getcwd()).resolve() == _our_parent
    except OSError:
        return False


def _expose_hf_tokenizers() -> None:
    # Search every sys.path entry EXCEPT the project root, so the finder picks
    # the real HuggingFace tokenizers in site-packages rather than this package.
    search_paths = [p for p in sys.path if not _resolves_to_parent(p)]

    spec = importlib.machinery.PathFinder.find_spec("tokenizers", search_paths)
    if spec is None or spec.origin is None or spec.loader is None:
        return  # HuggingFace tokenizers not installed — nothing to re-export

    # Safety: never re-select ourselves.
    if Path(spec.origin).resolve().parent == _our_dir:
        return

    saved_self = sys.modules.pop("tokenizers", None)
    try:
        hf = importlib.util.module_from_spec(spec)
        # Register under the canonical name BEFORE exec so HF's own
        # `from tokenizers.X import ...` submodule imports resolve to HF.
        sys.modules["tokenizers"] = hf
        spec.loader.exec_module(hf)

        if saved_self is not None:
            # Copy HF's public top-level names (AddedToken, Tokenizer, models,
            # pre_tokenizers, …) into our namespace; keep our own dunders/__path__
            # so our local submodules (event_tokenizer, remi_wrapper) stay importable.
            saved_self.__dict__.update(
                {k: v for k, v in vars(hf).items() if not k.startswith("__")}
            )
    finally:
        # Restore ourselves as the authoritative "tokenizers" module. HF's
        # submodules remain cached as tokenizers.* in sys.modules.
        if saved_self is not None:
            sys.modules["tokenizers"] = saved_self


_expose_hf_tokenizers()
