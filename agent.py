"""Standard CABT / Kaggle Playground entrypoint `agent(obs_dict: dict) -> list[int]`.

Loads the trained imitation learning checkpoint (`checkpoints/best_model.pt`)
and guarantees that every returned action is a valid `list[int]` of option indices
within `[minCount, maxCount]` (or the 60-card deck when `select` is None).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.features import int_or
from src.model import PTCGPolicyNet
from src.train import load_default_deck, load_policy_from_checkpoint

_MODEL: PTCGPolicyNet | None = None
_DEFAULT_DECK: list[int] | None = None
_CURRENT_CKPT: str | None = None


def load_agent_checkpoint(checkpoint_path: str | Path | None = None) -> None:
    """Explicitly load or reload a policy checkpoint for `agent(obs_dict)`."""
    global _MODEL, _DEFAULT_DECK, _CURRENT_CKPT
    ckpt_str = str(checkpoint_path or os.environ.get("PTCG_CHECKPOINT", "checkpoints/best_model.pt"))
    p = Path(ckpt_str)
    if p.exists():
        _MODEL, _DEFAULT_DECK, _ = load_policy_from_checkpoint(p, device="cpu")
        _CURRENT_CKPT = str(p.resolve())
    else:
        _MODEL = PTCGPolicyNet()
        _MODEL.eval()
        _DEFAULT_DECK = load_default_deck("deck.csv")
        _CURRENT_CKPT = "untrained_fallback"


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """CABT simulator agent entrypoint.

    Args:
        obs_dict: Observation dictionary provided by `cabt` / `cg.game`.

    Returns:
        list[int]: Selected legal option indices (or 60 card IDs during deck selection).
    """
    global _MODEL, _DEFAULT_DECK
    if _MODEL is None or _DEFAULT_DECK is None:
        load_agent_checkpoint()
    assert _MODEL is not None and _DEFAULT_DECK is not None

    if not isinstance(obs_dict, dict):
        return list(_DEFAULT_DECK)

    # Unwrap if kaggle_environments wraps inside `obs_dict["observation"]`
    inner = obs_dict
    if "select" not in inner and isinstance(inner.get("observation"), dict):
        inner = inner["observation"]

    sel = inner.get("select")
    cur = inner.get("current")
    if sel is None or cur is None:
        return list(_DEFAULT_DECK)

    opts = sel.get("option") or []
    n_opts = len(opts)
    raw_min = int_or(sel, "minCount", 0)
    raw_max = int_or(sel, "maxCount", 1)
    if n_opts == 0 or raw_max <= 0:
        return []

    min_c = max(0, min(raw_min, n_opts))
    max_c = max(min_c, min(raw_max, n_opts))

    try:
        act = _MODEL.select_action(inner, default_deck=_DEFAULT_DECK, device="cpu")
        if (
            isinstance(act, list)
            and all(isinstance(i, int) and 0 <= i < n_opts for i in act)
            and (min_c <= len(act) <= max_c)
            and len(set(act)) == len(act)
        ):
            return act
    except Exception:
        pass

    return list(range(min_c))
