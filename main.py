# main.py에 대한 예시 파일 안내입니다!

# import torch
# import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Move:
    x: int
    y: int
    z: int  # 0: horizontal, 1: vertical


def bitmap_to_edges(board_lines: Sequence, xsize: int, ysize: int) -> Set[Move]:
    edges: Set[Move] = set()
    for x in range(xsize + 1):
        for y in range(ysize + 1):
            cell = board_lines[x][y]
            if 0 <= x < xsize and cell[0]:
                edges.add(Move(x, y, 0))
            if 0 <= y < ysize and cell[1]:
                edges.add(Move(x, y, 1))
    return edges


class DotsAndBoxesBoard:
    def __init__(self, xsize: int, ysize: int, edges: Set[Move]):
        self.xsize = xsize
        self.ysize = ysize
        self.edges = edges

    def available_moves(self) -> List[Move]:
        moves: List[Move] = []
        for x in range(self.xsize):
            for y in range(self.ysize + 1):
                move = Move(x, y, 0)
                if move not in self.edges:
                    moves.append(move)
        for x in range(self.xsize + 1):
            for y in range(self.ysize):
                move = Move(x, y, 1)
                if move not in self.edges:
                    moves.append(move)
        return moves

    def adjacent_squares(self, move: Move) -> List[Tuple[int, int]]:
        squares: List[Tuple[int, int]] = []
        if move.z == 0:
            if move.y > 0:
                squares.append((move.x, move.y - 1))
            if move.y < self.ysize:
                squares.append((move.x, move.y))
        else:
            if move.x > 0:
                squares.append((move.x - 1, move.y))
            if move.x < self.xsize:
                squares.append((move.x, move.y))
        return squares

    def count_edges_of_square(self, square: Tuple[int, int]) -> int:
        sx, sy = square
        edges = (
            Move(sx, sy, 0),
            Move(sx, sy + 1, 0),
            Move(sx, sy, 1),
            Move(sx + 1, sy, 1),
        )
        return sum(1 for edge in edges if edge in self.edges)

    def boxes_completed_by_move(self, move: Move) -> int:
        completed = 0
        for square in self.adjacent_squares(move):
            if self.count_edges_of_square(square) == 3:
                completed += 1
        return completed

    def danger_score(self, move: Move) -> int:
        danger = 0
        for square in self.adjacent_squares(move):
            if self.count_edges_of_square(square) == 2:
                danger += 1
        return danger


class HeuristicAgent:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_move(self, board_lines: Sequence, xsize: int, ysize: int) -> Move:
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)
        moves = board.available_moves()
        if not moves:
            raise ValueError("더 이상 둘 수 있는 선이 없습니다.")

        self._rng.shuffle(moves)

        closing_moves = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if closing_moves:
            closing_moves.sort(key=lambda m: (-board.boxes_completed_by_move(m), self._rng.random()))
            return closing_moves[0]

        safe_moves = [m for m in moves if board.danger_score(m) == 0]
        if safe_moves:
            return self._rng.choice(safe_moves)

        moves.sort(key=lambda m: (board.danger_score(m), self._rng.random()))
        return moves[0]


model: Optional[HeuristicAgent] = None


def init():
    global model
    model = HeuristicAgent()


def run(board_lines, xsize, ysize):
    if model is None:
        init()
    move = model.select_move(board_lines, xsize, ysize)
    return [move.x, move.y, move.z]
