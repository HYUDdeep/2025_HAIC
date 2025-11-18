# main.py에 대한 예시 파일 안내입니다!

# import torch
# import os
import random
from dataclasses import dataclass

model = None  # 전역 변수로 모델을 유지해주세요! 변수 이름도 model로 유지하셔야 합니다!

# 사용할 모델의 구조를 지정하는 클래스나 사용할 보조 함수를 여기에 작성하세요.

# ========== Midgame-Rebuild Allcock Agent (A 버전) ==========
"""
목표:
- 샘플 AI에게 1승 19패 나오던 구조를 완전히 뜯어고친 중반전 전략.
- 핵심 아이디어:
  1) safe move 정의를 현실적으로 완화 (새로운 3-edge만 금지)
  2) midgame에서 "local chain risk" 기반으로 수를 고른다
     - move 주변 박스의 edge 변화만 집중 분석
  3) 점수 나는 수(scoring move)는 가능한 한 적극적으로 먹되,
     그래도 여러 scoring move 중 더 안전한 걸 선택
  4) safe move가 전혀 없을 때만 Allcock 엔드게임 이론 사용
"""

@dataclass(frozen=True)
class Move:
    x: int
    y: int
    z: int  # 0: horizontal, 1: vertical


def bitmap_to_edges(board_lines, xsize, ysize):
    """board_lines를 Move 집합으로 변환"""
    edges = set()
    for x in range(xsize + 1):
        for y in range(ysize + 1):
            cell = board_lines[x][y]
            if 0 <= x < xsize and cell[0]:
                edges.add(Move(x, y, 0))
            if 0 <= y < ysize and cell[1]:
                edges.add(Move(x, y, 1))
    return edges


class DotsAndBoxesBoard:
    def __init__(self, xsize, ysize, edges):
        self.xsize = xsize
        self.ysize = ysize
        self.edges = edges

    def available_moves(self):
        moves = []
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

    def adjacent_squares(self, move):
        squares = []
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
        completed = 0
        for square in self.adjacent_squares(move):
            if self.count_edges_of_square(square) == 3:
                completed += 1
        return completed

    def danger_score(self, move):
        """이 수를 두었을 때 생성되는 3-edge 박스의 개수 (위험도)"""
        danger = 0
        for square in self.adjacent_squares(move):
            if self.count_edges_of_square(square) == 2:
                danger += 1
        return danger


# ---------- 엔드게임용 컴포넌트 구조 ----------

@dataclass
class Component:
    """엔드게임에서의 한 컴포넌트 (long chain 또는 loop)."""
    boxes = None      # (x, y) 박스 좌표들
    length = 0        # 박스 개수
    is_loop = False   # 루프인지 여부
    missing_edges = None  # 이 컴포넌트 안에서 아직 안 그어진 선들


# ---------- 기본 유틸 함수들 ----------

def _count_box_edges(board, box):
    """해당 박스에 이미 그어진 선의 개수."""
    return board.count_edges_of_square(box)


def _is_safe_move(board, move):
    """
    "safe move" 정의 (완화된 버전):

    - move를 둔 이후, 어떤 박스도
      '새로운' 3-edge 상태가 되지 않으면 safe.

    즉, 기존에 2-edge였던 박스가 move 이후 3-edge가 되는 것은 금지.
    (이미 3-edge였던 건 우리가 만든 게 아니므로 허용)
    """
    tmp_edges = board.edges.copy()
    tmp_edges.add(move)
    tmp_board = DotsAndBoxesBoard(board.xsize, board.ysize, tmp_edges)

    for sq in board.adjacent_squares(move):
        before = board.count_edges_of_square(sq)
        after = tmp_board.count_edges_of_square(sq)
        if before < 3 and after == 3:
            # 우리가 새로 3-edge를 만든 경우 → 위험
            return False
    return True


def _local_struct_score(board, move):
    """
    move 주변(local) 구조를 평가하는 점수.

    기준:
      - 0→1 edge : 매우 안전, +4
      - 1→2 edge : 약간 위험 증가지만 아직 괜찮음, +2
      - 2→3 edge : chain 씨앗에서 3-edge로 직행, 강하게 -8
      - 그 외 변화는 크게 보지 않음

    + 중앙에 가까울수록 가산(거리 감소).
    """
    adj = list(board.adjacent_squares(move))
    if not adj:
        # 인접 박스가 없으면 그냥 중앙성만 봄
        adj = []

    tmp_edges = board.edges.copy()
    tmp_edges.add(move)
    tmp_board = DotsAndBoxesBoard(board.xsize, board.ysize, tmp_edges)

    score = 0.0

    for sq in adj:
        before = board.count_edges_of_square(sq)
        after = tmp_board.count_edges_of_square(sq)

        if before == 0 and after == 1:
            score += 4.0
        elif before == 1 and after == 2:
            score += 2.0
        elif before == 2 and after == 3:
            score -= 8.0

    # 중앙성 점수 추가 (선분의 "중간 지점" 기준)
    cx = board.xsize / 2.0
    cy = board.ysize / 2.0
    if move.z == 0:  # 가로선
        mx = move.x + 0.5
        my = move.y
    else:            # 세로선
        mx = move.x
        my = move.y + 0.5

    dist = abs(mx - cx) + abs(my - cy)
    score -= dist  # 중앙 가까울수록 더 좋은 점수

    return score


def _choose_best_safe(rng, board, moves):
    """
    여러 safe move 중에서:
      - local_struct_score가 가장 높은 수 선택
      - 동점이면 랜덤하게 tie-break
    """
    best = None
    best_score = float("-inf")

    for m in moves:
        s = _local_struct_score(board, m)
        # tie-break를 위해 약간의 랜덤 노이즈 추가
        s += rng.random() * 0.01
        if s > best_score:
            best = m
            best_score = s

    return best if best is not None else rng.choice(moves)


# ---------- 엔드게임: Allcock 이론 구현 ----------

def _build_components(board):
    """
    현재 보드에서 아직 완성되지 않은 박스들 중 연결된 컴포넌트를 찾는다.
    이때 연결 기준은 "공유하는 미완성 선(undrawn edge)".
    - 순수 2-edge 내부 구조는 loop
    - 그 외는 chain / 혼합 구조로 취급
    """
    xs, ys = board.xsize, board.ysize

    # 아직 완성되지 않은 박스들
    unclaimed = [
        (x, y)
        for x in range(xs)
        for y in range(ys)
        if board.count_edges_of_square((x, y)) < 4
    ]

    visited = set()
    components = []

    def neighbors_by_undrawn_edge(bx, by):
        res = []
        # 오른쪽 이웃
        if bx + 1 < xs:
            shared = Move(bx + 1, by, 1)  # 세로선
            if shared not in board.edges:
                res.append((bx + 1, by))
        # 아래 이웃
        if by + 1 < ys:
            shared = Move(bx, by + 1, 0)  # 가로선
            if shared not in board.edges:
                res.append((bx, by + 1))
        # 왼쪽 이웃
        if bx - 1 >= 0:
            shared = Move(bx, by, 1)
            if shared not in board.edges:
                res.append((bx - 1, by))
        # 위 이웃
        if by - 1 >= 0:
            shared = Move(bx, by, 0)
            if shared not in board.edges:
                res.append((bx, by - 1))
        return res

    for start in unclaimed:
        if start in visited:
            continue

        comp_boxes = []
        stack = [start]
        visited.add(start)

        while stack:
            b = stack.pop()
            comp_boxes.append(b)
            for nb in neighbors_by_undrawn_edge(b[0], b[1]):
                if nb not in visited and nb in unclaimed:
                    visited.add(nb)
                    stack.append(nb)

        if not comp_boxes:
            continue

        degrees = [_count_box_edges(board, b) for b in comp_boxes]
        # loop: 내부 박스 모두 2-edge
        is_loop = all(d == 2 for d in degrees)

        missing_edges_set = set()
        for (x, y) in comp_boxes:
            edges = [
                Move(x, y, 0),
                Move(x, y + 1, 0),
                Move(x, y, 1),
                Move(x + 1, y, 1),
            ]
            for e in edges:
                if e not in board.edges:
                    missing_edges_set.add(e)

        comp = Component()
        comp.boxes = comp_boxes
        comp.length = len(comp_boxes)
        comp.is_loop = is_loop
        comp.missing_edges = list(missing_edges_set)
        components.append(comp)

    return components


def _compute_controlled_value_and_counts(components):
    """
    Allcock 논문에서의 controlled value c(G) 및 통계 계산.

    반환: (c, size, theta, num_loops, num_4loops)
      - c: controlled value c(G)
      - size: 전체 박스 개수
      - theta: 3-chain 개수
      - num_loops: loop 개수
      - num_4loops: 길이 4 loop 개수
    """
    size = sum(c.length for c in components)
    chains = [c for c in components if not c.is_loop]
    loops = [c for c in components if c.is_loop]

    long_chains = [c for c in chains if c.length >= 3]
    theta = sum(1 for c in chains if c.length == 3)
    num_loops = len(loops)
    num_4loops = sum(1 for c in loops if c.length == 4)

    # terminal bonus tb(G)
    if size == 0:
        tb = 0
    elif num_loops > 0 and theta == 0:
        tb = 8  # loops only
    elif num_loops > 0 and theta > 0:
        tb = 6  # loops + 3-chains
    else:
        tb = 4  # 그 외

    c = size - 4 * len(long_chains) - 8 * num_loops + tb
    return c, size, theta, num_loops, num_4loops


def _select_allcock_opener_move(board, rng, moves):
    """
    엔드게임에서 Allcock opener 전략 적용.
    - G: loops + long chains 컴포넌트들의 집합
    - Theorem 1.1의 세 가지 케이스 + standard move 구현
    """
    components = _build_components(board)
    if not components:
        return None

    c, size, theta, num_loops, num_4loops = _compute_controlled_value_and_counts(components)

    chains = [cpt for cpt in components if not cpt.is_loop]
    loops = [cpt for cpt in components if cpt.is_loop]

    def pick_from_component(cpt_list, length_filter=None):
        if not cpt_list:
            return None
        if length_filter is not None:
            cpt_list = [c for c in cpt_list if c.length == length_filter]
            if not cpt_list:
                return None
        best = min(cpt_list, key=lambda cpt: cpt.length)
        legal = [m for m in best.missing_edges if m in moves]
        if not legal:
            return None
        return rng.choice(legal)

    def standard_move():
        """
        Allcock standard move:
          1) 3-chain 있으면 3-chain 열기
          2) 아니면 shortest loop 열기
          3) 아니면 shortest chain 열기
        """
        three_chains = [cpt for cpt in chains if cpt.length == 3]
        if three_chains:
            mv = pick_from_component(three_chains)
            if mv:
                return mv
        if loops:
            mv = pick_from_component(loops)
            if mv:
                return mv
        if chains:
            return pick_from_component(chains)
        return None

    # (1) c(G) ≥ 2 and G = 3 + (one or more loops)
    #     → 3-chain 하나 + 나머지 loops → loop를 먼저 연다
    if c >= 2 and theta == 1 and len(chains) == 1 and num_loops >= 1:
        mv = pick_from_component(loops)
        if mv:
            return mv

    # (2) c(G) ∈ {0, ±1} and G = 4` + (anything except 3+3+3)
    #     → 4-loop 하나 + 기타 → 4-loop를 먼저 연다 (단, 3+3+3+4 예외)
    if c in {0, 1, -1} and num_4loops > 0:
        is_3_3_3_4 = (
            theta == 3
            and num_4loops == 1
            and len(loops) == 1
            and all(cpt.length == 3 for cpt in chains)
        )
        if not is_3_3_3_4:
            mv = pick_from_component(loops, length_filter=4)
            if mv:
                return mv

    # (3) c(G) ≤ −2 and G = 4` + 3 + H, 4 | size(H), H에 3-chain 없음
    #     → 4-loop 먼저 연다
    if c <= -2 and num_4loops > 0 and theta == 1:
        size_H = size - 7  # 4-loop(4) + 3-chain(3)
        if size_H % 4 == 0:
            mv = pick_from_component(loops, length_filter=4)
            if mv:
                return mv

    # 그 외 모든 경우 standard move가 optimal (Allcock Theorem 1.1)
    mv = standard_move()
    if mv:
        return mv

    # 혹시라도 실패하면 그냥 랜덤 합법 수
    return rng.choice(moves) if moves else None


# ---------- 메인 에이전트 ----------

class AllcockAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_move(self, board_lines, xsize, ysize):
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)
        moves = board.available_moves()

        if not moves:
            # 둘 수 있는 수가 아예 없다면 (이론상 거의 없음)
            return Move(0, 0, 0)

        # 1. 점수 나는 수(scoring move)가 있으면 → 적극적으로 먹되,
        #    여러 수 중 local 구조가 더 좋은 것을 고른다.
        scoring_moves = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if scoring_moves:
            def scoring_key(m):
                boxes = board.boxes_completed_by_move(m)
                struct = _local_struct_score(board, m)
                # 박스 개수 우선, 그 다음 local 구조
                return (boxes, struct)

            best_scoring = max(scoring_moves, key=scoring_key)
            return best_scoring

        # 2. safe move(새로운 3-edge를 만들지 않는 수)가 있다면
        #    → midgame 전략: local_struct_score 기반으로 가장 좋은 safe 선택
        safe_moves = [m for m in moves if _is_safe_move(board, m)]
        if safe_moves:
            return _choose_best_safe(self.rng, board, safe_moves)

        # 3. safe move가 전혀 없다면 → 진짜 엔드게임이라고 보고
        #    Allcock 엔드게임 opener 전략 적용
        end_move = _select_allcock_opener_move(board, self.rng, moves)
        if end_move is not None:
            return end_move

        # 4. 방어 코드: 혹시 위에서 실패하면 랜덤
        return self.rng.choice(moves)


# 반드시 init(),run()함수를 구현해줘야 합니다. 없으면 에러가 발생합니다.
def init():
    # << 체점 시 양쪽 에이전트에 대해서 처음 한 번 실행되는 함수입니다. >>
    global model
    
    # Midgame-Rebuild Allcock 에이전트 초기화
    model = AllcockAgent()


def run(board_lines, xsize, ysize):
    # << 에이전트의 차례가 될 때마다 실행되는 함수입니다. >>
    global model
    
    if model is None:
        init()
    
    # AllcockAgent로 최적의 수 선택
    move = model.select_move(board_lines, xsize, ysize)
    
    # [x, y, z] 형태로 반환
    return [move.x, move.y, move.z]