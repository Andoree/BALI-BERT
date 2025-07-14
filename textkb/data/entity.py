from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    mention_str: str
    span_start: int
    span_end: int
    node_ids: List



@dataclass
class Token:
    text: str