# -*- coding: utf-8 -*-
"""
JSON schema helpers for NS-3 <-> Python messages.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class StateMsg:
    type: str
    ue: int
    size: int
    cycles: int

@dataclass
class ActionMsg:
    type: str
    ue: int
    action: int  # 0=local, 1=offload

@dataclass
class MetricsMsg:
    type: str
    round: int
    comm_bytes: int
    avg_latency_ms: float
    avg_energy_j: float

def to_json(d: Dict[str, Any]) -> str:
    import json
    return json.dumps(d, separators=(",", ":"))

def dataclass_to_json(obj) -> str:
    return to_json(asdict(obj))
