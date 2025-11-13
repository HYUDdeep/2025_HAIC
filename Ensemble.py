"""
Ensemble Agent for Dots and Boxes
다양한 전략을 결합하여 최적의 수를 선택하는 앙상블 에이전트
"""

import random
import time
from typing import List, Optional, Sequence, Set, Tuple, Dict
from dataclasses import dataclass
from collections import Counter

from main import Move, DotsAndBoxesBoard, bitmap_to_edges


@dataclass
class GameState:
    """게임 상태를 표현하는 클래스"""
    edges: Set[Move]
    scores: Tuple[int, int]
    current_player: int
    xsize: int
    ysize: int
    
    def copy(self):
        return GameState(
            edges=self.edges.copy(),
            scores=self.scores,
            current_player=self.current_player,
            xsize=self.xsize,
            ysize=self.ysize
        )


class EnsembleAgent:
    """
    하이브리드 앙상블 에이전트
    
    전략:
    1. 빠른 얕은 탐색으로 유망한 후보 수들을 선정 (0.2초)
    2. 선정된 후보들만 깊게 재탐색 (0.6초)
    3. 다양한 평가 함수로 검증
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        time_limit: float = 0.8
    ):
        self._rng = random.Random(seed)
        self.time_limit = time_limit
        self.nodes_searched = 0
        self.transposition_table: Dict[frozenset, Tuple[int, float]] = {}
        self.start_time = 0.0
        
        # 통계
        self.total_moves = 0
        self.strategy_votes = {"shallow": 0, "deep": 0, "consensus": 0}
    
    def select_move(self, board_lines: Sequence, xsize: int, ysize: int) -> Move:
        """앙상블 전략으로 최적의 수 선택"""
        self.start_time = time.time()
        self.nodes_searched = 0
        
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)
        moves = board.available_moves()
        
        if not moves:
            raise ValueError("더 이상 둘 수 있는 선이 없습니다.")
        
        if len(moves) == 1:
            return moves[0]
        
        # 즉시 박스를 완성할 수 있는 수가 있으면 우선 고려
        immediate_win_moves = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if immediate_win_moves:
            # 여러 개면 가장 많이 완성하는 수 선택
            immediate_win_moves.sort(key=lambda m: -board.boxes_completed_by_move(m))
            # 게임 초반이면 바로 반환 (시간 절약)
            if len(moves) > 30:
                return immediate_win_moves[0]
        
        # 게임 진행도 파악
        total_edges = xsize * (ysize + 1) + ysize * (xsize + 1)
        progress = (total_edges - len(moves)) / total_edges
        
        # 게임 상태 생성
        state = GameState(
            edges=edges.copy(),
            scores=(0, 0),
            current_player=0,
            xsize=xsize,
            ysize=ysize
        )
        
        # 전략 선택
        if len(moves) > 35:
            # 게임 초반: 빠른 휴리스틱만 사용
            best_move = self._quick_heuristic_selection(board, moves)
            self.strategy_votes["shallow"] += 1
        elif len(moves) > 15:
            # 게임 중반: 2단계 앙상블
            best_move = self._two_stage_ensemble(state, board, moves)
            self.strategy_votes["consensus"] += 1
        else:
            # 게임 후반: 깊은 탐색
            best_move = self._deep_search(state, board, moves, depth=5)
            self.strategy_votes["deep"] += 1
        
        self.total_moves += 1
        return best_move
    
    def _quick_heuristic_selection(self, board: DotsAndBoxesBoard, moves: List[Move]) -> Move:
        """빠른 휴리스틱 기반 선택"""
        self._rng.shuffle(moves)
        
        # 1. 박스 완성 수
        closing_moves = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if closing_moves:
            return max(closing_moves, key=lambda m: board.boxes_completed_by_move(m))
        
        # 2. 안전한 수 중 선택
        safe_moves = [m for m in moves if board.danger_score(m) == 0]
        if safe_moves:
            return self._rng.choice(safe_moves)
        
        # 3. 가장 덜 위험한 수
        return min(moves, key=lambda m: board.danger_score(m))
    
    def _two_stage_ensemble(self, state: GameState, board: DotsAndBoxesBoard, moves: List[Move]) -> Move:
        """
        2단계 앙상블:
        Stage 1: 얕은 탐색으로 후보 선정
        Stage 2: 후보들만 깊게 재탐색
        """
        time_for_stage1 = self.time_limit * 0.25  # 25% 시간
        time_for_stage2 = self.time_limit * 0.65  # 65% 시간
        
        # Stage 1: 얕은 탐색 (depth=2)으로 후보 선정
        candidates = []
        stage1_deadline = self.start_time + time_for_stage1
        
        moves = self._order_moves(board, moves)
        
        for move in moves[:min(len(moves), 10)]:  # 최대 10개만 평가
            if time.time() > stage1_deadline:
                break
            
            try:
                new_state, boxes = self._apply_move(state, board, move)
                score = self._minimax(
                    new_state, 
                    depth=2, 
                    alpha=float('-inf'), 
                    beta=float('inf'), 
                    maximizing=(boxes > 0),
                    deadline=stage1_deadline
                )
                candidates.append((move, score))
            except TimeoutError:
                break
        
        if not candidates:
            return self._quick_heuristic_selection(board, moves)
        
        # 상위 3개 후보 선정
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [m for m, _ in candidates[:3]]
        
        # Stage 2: 후보들만 깊게 재탐색 (depth=4)
        stage2_deadline = self.start_time + time_for_stage1 + time_for_stage2
        
        best_move = None
        best_score = float('-inf')
        
        for move in top_candidates:
            if time.time() > stage2_deadline:
                break
            
            try:
                new_state, boxes = self._apply_move(state, board, move)
                score = self._minimax(
                    new_state,
                    depth=4,
                    alpha=float('-inf'),
                    beta=float('inf'),
                    maximizing=(boxes > 0),
                    deadline=stage2_deadline
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move
            except TimeoutError:
                break
        
        return best_move if best_move else top_candidates[0]
    
    def _deep_search(self, state: GameState, board: DotsAndBoxesBoard, moves: List[Move], depth: int) -> Move:
        """깊은 탐색"""
        deadline = self.start_time + self.time_limit * 0.9
        
        moves = self._order_moves(board, moves)
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in moves:
            if time.time() > deadline:
                break
            
            try:
                new_state, boxes = self._apply_move(state, board, move)
                score = self._minimax(
                    new_state,
                    depth=depth,
                    alpha=alpha,
                    beta=beta,
                    maximizing=(boxes > 0),
                    deadline=deadline
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
            except TimeoutError:
                break
        
        return best_move if best_move else moves[0]
    
    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        deadline: float
    ) -> float:
        """MinMax with Alpha-Beta Pruning"""
        
        self.nodes_searched += 1
        
        # 시간 체크
        if self.nodes_searched % 1000 == 0:
            if time.time() > deadline:
                raise TimeoutError()
        
        # Transposition Table
        state_key = frozenset(state.edges)
        if state_key in self.transposition_table:
            cached_depth, cached_score = self.transposition_table[state_key]
            if cached_depth >= depth:
                return cached_score
        
        board = DotsAndBoxesBoard(state.xsize, state.ysize, state.edges)
        moves = board.available_moves()
        
        # 종료 조건
        if depth == 0 or not moves:
            score = self._evaluate_state(state, board, len(moves))
            self.transposition_table[state_key] = (depth, score)
            return score
        
        moves = self._order_moves(board, moves)
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                new_state, boxes = self._apply_move(state, board, move)
                eval_score = self._minimax(
                    new_state,
                    depth - 1,
                    alpha,
                    beta,
                    boxes > 0,
                    deadline
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            self.transposition_table[state_key] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_state, boxes = self._apply_move(state, board, move)
                eval_score = self._minimax(
                    new_state,
                    depth - 1,
                    alpha,
                    beta,
                    boxes == 0,
                    deadline
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            self.transposition_table[state_key] = (depth, min_eval)
            return min_eval
    
    def _apply_move(self, state: GameState, board: DotsAndBoxesBoard, move: Move) -> Tuple[GameState, int]:
        """수를 적용하고 새로운 상태 반환"""
        new_state = state.copy()
        new_state.edges.add(move)
        
        boxes_completed = board.boxes_completed_by_move(move)
        
        if boxes_completed > 0:
            if state.current_player == 0:
                new_state.scores = (state.scores[0] + boxes_completed, state.scores[1])
            else:
                new_state.scores = (state.scores[0], state.scores[1] + boxes_completed)
        else:
            new_state.current_player = 1 - state.current_player
        
        return new_state, boxes_completed
    
    def _evaluate_state(self, state: GameState, board: DotsAndBoxesBoard, remaining_moves: int) -> float:
        """향상된 평가 함수"""
        score_diff = state.scores[0] - state.scores[1]
        
        # 게임 종료
        if remaining_moves == 0:
            if score_diff > 0:
                return 10000 + score_diff * 100
            elif score_diff < 0:
                return -10000 + score_diff * 100
            else:
                return 0
        
        eval_score = score_diff * 100
        
        # 게임 후반일수록 점수 차이가 더 중요
        if remaining_moves < 10:
            eval_score = score_diff * 200
        
        moves = board.available_moves()
        
        # 즉시 획득 가능한 박스
        immediate_boxes = sum(board.boxes_completed_by_move(m) for m in moves)
        if state.current_player == 0:
            eval_score += immediate_boxes * 80
        else:
            eval_score -= immediate_boxes * 80
        
        # 안전한 수의 개수
        safe_moves = sum(1 for m in moves if board.danger_score(m) == 0)
        eval_score += safe_moves * 5
        
        # 위험한 박스들
        dangerous_boxes = 0
        for x in range(state.xsize):
            for y in range(state.ysize):
                edges_count = board.count_edges_of_square((x, y))
                if edges_count == 2:
                    dangerous_boxes += 1
                elif edges_count == 3:
                    dangerous_boxes += 3
        
        eval_score -= dangerous_boxes * 8
        
        return eval_score
    
    def _order_moves(self, board: DotsAndBoxesBoard, moves: List[Move]) -> List[Move]:
        """향상된 무브 오더링"""
        def move_priority(move: Move) -> Tuple[int, int, int, float]:
            boxes = board.boxes_completed_by_move(move)
            
            # 3개 선 박스 확인
            adjacent_three = 0
            for square in board.adjacent_squares(move):
                if board.count_edges_of_square(square) == 3:
                    adjacent_three += 1
            
            danger = board.danger_score(move)
            random_tie = self._rng.random()
            
            return (-boxes, adjacent_three, danger, random_tie)
        
        return sorted(moves, key=move_priority)
    
    def print_stats(self):
        """통계 출력"""
        if self.total_moves > 0:
            print(f"\n📊 Ensemble 전략 사용 통계 (총 {self.total_moves}수):")
            for strategy, count in self.strategy_votes.items():
                percentage = (count / self.total_moves) * 100
                print(f"  {strategy}: {count}회 ({percentage:.1f}%)")


# main.py와 호환되는 인터페이스
model: Optional[EnsembleAgent] = None


def init():
    global model
    model = EnsembleAgent(time_limit=0.8)


def run(board_lines, xsize, ysize):
    if model is None:
        init()
    move = model.select_move(board_lines, xsize, ysize)
    return [move.x, move.y, move.z]

