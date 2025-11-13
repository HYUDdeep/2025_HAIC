"""
MinMax Agent with Alpha-Beta Pruning for Dots and Boxes
6x6 dots (5x5 boxes) 게임에 최적화된 구현
"""

import random
import time
from typing import List, Optional, Sequence, Set, Tuple, Dict
from dataclasses import dataclass

from main import Move, DotsAndBoxesBoard, bitmap_to_edges


@dataclass
class GameState:
    """게임 상태를 표현하는 클래스"""
    edges: Set[Move]
    scores: Tuple[int, int]  # (player0, player1)
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


class MinMaxAgent:
    """MinMax Algorithm with Alpha-Beta Pruning"""
    
    def __init__(
        self, 
        seed: Optional[int] = None,
        max_depth: int = 4,
        time_limit: float = 0.9  # 초당 1수로 가정, 여유를 두고 0.9초
    ):
        self._rng = random.Random(seed)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.nodes_searched = 0
        self.transposition_table: Dict[frozenset, Tuple[int, float, Optional[Move]]] = {}
        self.start_time = 0.0
        
    def select_move(self, board_lines: Sequence, xsize: int, ysize: int) -> Move:
        """최적의 수를 선택"""
        self.start_time = time.time()
        self.nodes_searched = 0
        
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)
        moves = board.available_moves()
        
        if not moves:
            raise ValueError("더 이상 둘 수 있는 선이 없습니다.")
        
        # 한 수만 남았으면 바로 반환
        if len(moves) == 1:
            return moves[0]
        
        # 게임 상태 생성
        state = GameState(
            edges=edges.copy(),
            scores=(0, 0),  # 현재까지의 점수는 보드에서 계산 가능
            current_player=0,  # 항상 현재 플레이어 관점에서
            xsize=xsize,
            ysize=ysize
        )
        
        # 현재 점수 계산 (간단한 방법: 전체 박스 수 - 남은 박스 수)
        state.scores = self._calculate_current_scores(board, xsize, ysize)
        
        # Iterative Deepening: 시간이 허용하는 한 깊이를 늘려가며 탐색
        best_move = None
        best_score = float('-inf')
        
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.time_limit * 0.8:
                break
                
            try:
                move, score = self._iterative_search(state, board, depth)
                if move is not None:
                    best_move = move
                    best_score = score
            except TimeoutError:
                break
        
        if best_move is None:
            # 폴백: 휴리스틱 기반 선택
            best_move = self._fallback_selection(board, moves)
        
        return best_move
    
    def _iterative_search(self, state: GameState, board: DotsAndBoxesBoard, depth: int) -> Tuple[Optional[Move], float]:
        """주어진 깊이로 탐색"""
        moves = board.available_moves()
        
        # 무브 오더링: 더 유망한 수를 먼저 탐색
        moves = self._order_moves(board, moves)
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in moves:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
            
            # 수를 두고 평가
            new_state, boxes_completed = self._apply_move(state, board, move)
            
            if boxes_completed > 0:
                # 박스를 완성하면 같은 플레이어가 다시 둠
                score = self._minimax(new_state, depth - 1, alpha, beta, True)
            else:
                # 상대방 차례
                score = self._minimax(new_state, depth - 1, alpha, beta, False)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        return best_move, best_score
    
    def _minimax(
        self, 
        state: GameState, 
        depth: int, 
        alpha: float, 
        beta: float, 
        maximizing: bool
    ) -> float:
        """MinMax with Alpha-Beta Pruning"""
        
        self.nodes_searched += 1
        
        # 시간 체크
        if self.nodes_searched % 100 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
        
        # Transposition Table 체크
        state_key = frozenset(state.edges)
        if state_key in self.transposition_table:
            cached_depth, cached_score, _ = self.transposition_table[state_key]
            if cached_depth >= depth:
                return cached_score
        
        board = DotsAndBoxesBoard(state.xsize, state.ysize, state.edges)
        moves = board.available_moves()
        
        # 종료 조건: 깊이 0 또는 게임 종료
        if depth == 0 or not moves:
            score = self._evaluate_state(state, board)
            self.transposition_table[state_key] = (depth, score, None)
            return score
        
        # 무브 오더링
        moves = self._order_moves(board, moves)
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                new_state, boxes_completed = self._apply_move(state, board, move)
                
                if boxes_completed > 0:
                    # 같은 플레이어 계속
                    eval_score = self._minimax(new_state, depth - 1, alpha, beta, True)
                else:
                    # 상대방 차례
                    eval_score = self._minimax(new_state, depth - 1, alpha, beta, False)
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Beta cut-off
            
            self.transposition_table[state_key] = (depth, max_eval, None)
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_state, boxes_completed = self._apply_move(state, board, move)
                
                if boxes_completed > 0:
                    # 상대방 계속
                    eval_score = self._minimax(new_state, depth - 1, alpha, beta, False)
                else:
                    # 현재 플레이어 차례
                    eval_score = self._minimax(new_state, depth - 1, alpha, beta, True)
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha cut-off
            
            self.transposition_table[state_key] = (depth, min_eval, None)
            return min_eval
    
    def _apply_move(self, state: GameState, board: DotsAndBoxesBoard, move: Move) -> Tuple[GameState, int]:
        """수를 적용하고 새로운 상태 반환"""
        new_state = state.copy()
        new_state.edges.add(move)
        
        # 완성된 박스 수 계산
        boxes_completed = board.boxes_completed_by_move(move)
        
        if boxes_completed > 0:
            # 현재 플레이어가 점수 획득
            if state.current_player == 0:
                new_state.scores = (state.scores[0] + boxes_completed, state.scores[1])
            else:
                new_state.scores = (state.scores[0], state.scores[1] + boxes_completed)
        else:
            # 플레이어 교체
            new_state.current_player = 1 - state.current_player
        
        return new_state, boxes_completed
    
    def _evaluate_state(self, state: GameState, board: DotsAndBoxesBoard) -> float:
        """게임 상태 평가 함수"""
        # 기본: 점수 차이
        score_diff = state.scores[0] - state.scores[1]
        
        # 게임이 끝났으면 큰 보너스/페널티
        moves = board.available_moves()
        if not moves:
            if score_diff > 0:
                return 10000 + score_diff
            elif score_diff < 0:
                return -10000 + score_diff
            else:
                return 0
        
        # 휴리스틱 요소들
        eval_score = score_diff * 100
        
        # 안전한 수의 개수 (위험도 0인 수)
        safe_moves = sum(1 for m in moves if board.danger_score(m) == 0)
        eval_score += safe_moves * 5
        
        # 즉시 획득 가능한 박스 수
        immediate_boxes = sum(board.boxes_completed_by_move(m) for m in moves)
        if state.current_player == 0:
            eval_score += immediate_boxes * 50
        else:
            eval_score -= immediate_boxes * 50
        
        # 2개 선이 그어진 박스 수 (위험한 상태)
        dangerous_boxes = 0
        for x in range(state.xsize):
            for y in range(state.ysize):
                if board.count_edges_of_square((x, y)) == 2:
                    dangerous_boxes += 1
        eval_score -= dangerous_boxes * 10
        
        return eval_score
    
    def _order_moves(self, board: DotsAndBoxesBoard, moves: List[Move]) -> List[Move]:
        """무브 오더링: 더 유망한 수를 먼저 탐색"""
        def move_priority(move: Move) -> Tuple[int, int, float]:
            # 1. 박스를 완성하는 수 (가장 우선)
            boxes = board.boxes_completed_by_move(move)
            
            # 2. 위험도 (낮을수록 좋음)
            danger = board.danger_score(move)
            
            # 3. 랜덤 타이브레이커
            random_tie = self._rng.random()
            
            return (-boxes, danger, random_tie)
        
        return sorted(moves, key=move_priority)
    
    def _calculate_current_scores(self, board: DotsAndBoxesBoard, xsize: int, ysize: int) -> Tuple[int, int]:
        """현재까지의 점수를 계산 (실제로는 게임에서 추적해야 하지만 여기서는 간단히 0으로)"""
        # 실제 게임에서는 점수를 추적해야 하지만, 
        # select_move 호출 시점에서는 이전 점수를 알 수 없으므로 0으로 시작
        # MinMax는 상대적 점수 차이를 평가하므로 절대 점수는 중요하지 않음
        return (0, 0)
    
    def _fallback_selection(self, board: DotsAndBoxesBoard, moves: List[Move]) -> Move:
        """시간 초과 등의 경우 폴백 휴리스틱 선택"""
        self._rng.shuffle(moves)
        
        # 1. 박스를 완성하는 수
        closing_moves = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if closing_moves:
            closing_moves.sort(key=lambda m: (-board.boxes_completed_by_move(m), self._rng.random()))
            return closing_moves[0]
        
        # 2. 안전한 수 (위험도 0)
        safe_moves = [m for m in moves if board.danger_score(m) == 0]
        if safe_moves:
            return self._rng.choice(safe_moves)
        
        # 3. 가장 덜 위험한 수
        moves.sort(key=lambda m: (board.danger_score(m), self._rng.random()))
        return moves[0]


# main.py와 호환되는 인터페이스
model: Optional[MinMaxAgent] = None


def init():
    global model
    # 깊이 4로 시작 (조정 가능)
    model = MinMaxAgent(max_depth=4, time_limit=0.9)


def run(board_lines, xsize, ysize):
    if model is None:
        init()
    move = model.select_move(board_lines, xsize, ysize)
    return [move.x, move.y, move.z]

