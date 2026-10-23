"""Source-owned blocks; coordinates are Python Unicode character offsets."""
from dataclasses import asdict, dataclass


@dataclass
class Block:
    block_id: int
    start: int
    end: int
    text: str
    tokens: int
    embedding_tokens: int

    def to_dict(self):
        return asdict(self)
