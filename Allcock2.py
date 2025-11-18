import random
from dataclasses import dataclass

model = None  # 반드시 유지


# ==============================
# 기본 Move 자료형
# ==============================
@dataclass(frozen=True)
class Move:
    x: int
    y: int
    z: int  # 0: horizontal, 1: vertical


# ==============================
# Bitmap → Edge 변환
# ==============================
def bitmap_to_edges(board_lines, xsize, ysize):
    edges = set()
    for x in range(xsize + 1):
        for y in range(ysize + 1):
            cell = board_lines[x][y]
            # horizontal
            if 0 <= x < xsize and cell[0]:
                edges.add(Move(x, y, 0))
            # vertical
            if 0 <= y < ysize and cell[1]:
                edges.add(Move(x, y, 1))
    return edges


# ==============================
# Dots and Boxes Board
# ==============================
class DotsAndBoxesBoard:
    def __init__(self, xsize, ysize, edges):
        self.xsize = xsize
        self.ysize = ysize
        self.edges = edges

    # ====== A+ 안정화 버전: move 범위 clamp ======
    def available_moves(self):
        moves = []

        # horizontal: x=[0..xsize-1], y=[0..ysize]
        for x in range(self.xsize):
            for y in range(self.ysize + 1):
                if not (0 <= x < self.xsize and 0 <= y <= self.ysize):
                    continue
                mv = Move(x, y, 0)
                if mv not in self.edges:
                    moves.append(mv)

        # vertical: x=[0..xsize], y=[0..ysize-1]
        for x in range(self.xsize + 1):
            for y in range(self.ysize):
                if not (0 <= x <= self.xsize and 0 <= y < self.ysize):
                    continue
                mv = Move(x, y, 1)
                if mv not in self.edges:
                    moves.append(mv)

        return moves

    def adjacent_squares(self, move):
        squares = []
        if move.z == 0:
            # horizontal
            if move.y > 0:
                squares.append((move.x, move.y - 1))
            if move.y < self.ysize:
                squares.append((move.x, move.y))
        else:
            # vertical
            if move.x > 0:
                squares.append((move.x - 1, move.y))
            if move.x < self.xsize:
                squares.append((move.x, move.y))
        return squares

    def count_edges_of_square(self, square):
        sx, sy = square
        edges = (
            Move(sx, sy, 0),
            Move(sx, sy + 1, 0),
            Move(sx, sy, 1),
            Move(sx + 1, sy, 1),
        )
        return sum(1 for edge in edges if edge in self.edges)

    def boxes_completed_by_move(self, move):
        cnt = 0
        for sq in self.adjacent_squares(move):
            if self.count_edges_of_square(sq) == 3:
                cnt += 1
        return cnt

    def danger_score(self, move):
        # 생성 위험도
        danger = 0
        for sq in self.adjacent_squares(move):
            if self.count_edges_of_square(sq) == 2:
                danger += 1
        return danger


# ==============================
# Safe Move 검사
# ==============================
def _is_safe_move(board, move):
    temp_edges = board.edges.copy()
    temp_edges.add(move)
    new_board = DotsAndBoxesBoard(board.xsize, board.ysize, temp_edges)

    # 새로 3-edge 박스를 만드는지 검사
    for sq in board.adjacent_squares(move):
        before = board.count_edges_of_square(sq)
        after = new_board.count_edges_of_square(sq)
        if before < 3 and after == 3:
            return False
    return True


# ==============================
# Local structural score
# ==============================
def _local_struct_score(board, move):
    temp_edges = board.edges.copy()
    temp_edges.add(move)
    new_board = DotsAndBoxesBoard(board.xsize, board.ysize, temp_edges)

    score = 0
    for sq in board.adjacent_squares(move):
        before = board.count_edges_of_square(sq)
        after = new_board.count_edges_of_square(sq)
        if before == 0 and after == 1:
            score += 4
        elif before == 1 and after == 2:
            score += 2
        elif before == 2 and after == 3:
            score -= 8

    # 중심 거리 페널티 — clamp 강화
    cx, cy = board.xsize / 2, board.ysize / 2
    if move.z == 0:
        mx, my = move.x + 0.5, move.y
    else:
        mx, my = move.x, move.y + 0.5

    # 절대 범위 벗어나지 않도록 clamp
    mx = max(0, min(board.xsize, mx))
    my = max(0, min(board.ysize, my))

    score -= abs(mx - cx) + abs(my - cy)
    return score


def _choose_best_safe(rng, board, moves):
    best = None
    best_score = -999999
    for m in moves:
        s = _local_struct_score(board, m) + rng.random() * 0.01
        if s > best_score:
            best = m
            best_score = s
    return best if best else rng.choice(moves)


# ==============================
# 엔드게임 오프너 (안정화)
# ==============================
def _select_allcock_opener_move(board, rng, moves):
    # 실제 Allcock 계산은 여기 생략 (핵심만 사용)
    # → 안정화를 위해 가능 move 중 가장 "안전"한 것 선택
    safe = [m for m in moves if _is_safe_move(board, m)]
    if safe:
        return _choose_best_safe(rng, board, safe)

    # 없으면 위험도 최소
    best = min(moves, key=lambda m: board.danger_score(m))
    return best


# ==============================
# 메인 Agent
# ==============================
class AllcockAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_move(self, board_lines, xsize, ysize):
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)

        moves = board.available_moves()

        if not moves:
            return Move(0, 0, 0)

        # 1) 점수 나는 수 우선
        scoring = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if scoring:
            return max(scoring, key=lambda m: (board.boxes_completed_by_move(m), _local_struct_score(board, m)))

        # 2) safe move 존재 시 그 중 가장 좋은 수
        safe = [m for m in moves if _is_safe_move(board, m)]
        if safe:
            return _choose_best_safe(self.rng, board, safe)

        # 3) 엔드게임
        mv = _select_allcock_opener_move(board, self.rng, moves)
        if mv is not None and mv in moves:
            return mv

        # 4) 🚨 안정화 fallback: 여기서는 절대 invalid move 안 나옴
        legal_moves = board.available_moves()  # 재생성
        return self.rng.choice(legal_moves)


# ==============================
# API
# ==============================
def init():
    global model
    model = AllcockAgent()


def run(board_lines, xsize, ysize):
    global model
    if model is None:
        init()

    mv = model.select_move(board_lines, xsize, ysize)
    return [mv.x, mv.y, mv.z]
