import random
from typing import Optional, Sequence

from main import DotsAndBoxesBoard, Move, bitmap_to_edges


class RecklessAgent:
    """위험한 선을 우선적으로 두는 약한 에이전트."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_move(self, board_lines: Sequence, xsize: int, ysize: int) -> Move:
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)
        moves = board.available_moves()
        if not moves:
            raise ValueError("둘 수 있는 수가 없습니다.")

        self._rng.shuffle(moves)
        moves.sort(key=lambda m: (board.danger_score(m), self._rng.random()), reverse=True)
        return moves[0]
