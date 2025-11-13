"""
Dynamic Depth MinMax Agent for Dots and Boxes
게임 진행도에 따라 탐색 깊이를 동적으로 조정하여 속도와 정확도의 균형을 맞춤
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


class DynamicDepthMinMaxAgent:
    """
    동적 깊이 조정 MinMax 알고리즘
    - 게임 초반: 얕은 탐색 (빠른 속도)
    - 게임 중반: 중간 탐색 (균형)
    - 게임 후반: 깊은 탐색 (정확도)
    """
    
    def __init__(
        self, 
        seed: Optional[int] = None,
        time_limit: float = 0.8  # 안전 마진을 위해 0.8초
    ):
        self._rng = random.Random(seed)
        self.time_limit = time_limit
        self.nodes_searched = 0
        self.transposition_table: Dict[frozenset, Tuple[int, float, Optional[Move]]] = {}
        self.start_time = 0.0
        
        # 게임 통계 (디버깅용)
        self.total_moves = 0
        self.depth_usage = {2: 0, 3: 0, 4: 0, 5: 0}
        
    def _get_dynamic_depth(self, available_moves: int, total_edges: int) -> int:
        """
        남은 수에 따라 동적으로 탐색 깊이 결정
        
        Args:
            available_moves: 남은 가능한 수의 개수
            total_edges: 전체 선의 개수 (60개)
        
        Returns:
            적절한 탐색 깊이
        """
        # 게임 진행도 계산
        progress = (total_edges - available_moves) / total_edges
        
        # 게임 초반 (진행도 0-33%): 많은 선택지, 얕은 탐색
        if available_moves > 40:
            return 2
        
        # 게임 중반 (진행도 33-75%): 적당한 탐색
        elif available_moves > 15:
            return 3
        
        # 게임 후반 (진행도 75-90%): 깊은 탐색
        elif available_moves > 6:
            return 4
        
        # 게임 종반 (진행도 90-100%): 매우 깊은 탐색 또는 완전 탐색
        else:
            return 5
    
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
        
        # 전체 선의 개수 계산
        total_edges = xsize * (ysize + 1) + ysize * (xsize + 1)
        
        # 동적으로 깊이 결정
        dynamic_depth = self._get_dynamic_depth(len(moves), total_edges)
        self.depth_usage[dynamic_depth] = self.depth_usage.get(dynamic_depth, 0) + 1
        self.total_moves += 1
        
        # 즉시 박스를 완성할 수 있는 수가 있으면 우선 선택
        immediate_win_moves = [m for m in moves if board.boxes_completed_by_move(m) > 0]
        if immediate_win_moves:
            # 여러 개면 가장 많이 완성하는 수 선택
            immediate_win_moves.sort(key=lambda m: -board.boxes_completed_by_move(m))
            # 간단한 경우는 바로 반환 (시간 절약)
            if len(moves) > 20:  # 초반이면
                return immediate_win_moves[0]
        
        # 게임 상태 생성
        state = GameState(
            edges=edges.copy(),
            scores=(0, 0),
            current_player=0,
            xsize=xsize,
            ysize=ysize
        )
        
        # MinMax 탐색
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # 무브 오더링
        moves = self._order_moves(board, moves)
        
        for move in moves:
            if time.time() - self.start_time > self.time_limit * 0.9:
                break
            
            try:
                # 수를 두고 평가
                new_state, boxes_completed = self._apply_move(state, board, move)
                
                if boxes_completed > 0:
                    # 박스를 완성하면 같은 플레이어가 다시 둠
                    score = self._minimax(new_state, dynamic_depth - 1, alpha, beta, True)
                else:
                    # 상대방 차례
                    score = self._minimax(new_state, dynamic_depth - 1, alpha, beta, False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                
            except TimeoutError:
                break
        
        # 최선의 수를 찾지 못한 경우 폴백
        if best_move is None:
            best_move = self._fallback_selection(board, moves)
        
        return best_move
    
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
        
        # 시간 체크 (가끔씩만 - 성능을 위해)
        if self.nodes_searched % 1000 == 0:
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
            score = self._evaluate_state(state, board, len(moves))
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
    
    def _evaluate_state(self, state: GameState, board: DotsAndBoxesBoard, remaining_moves: int) -> float:
        """
        게임 상태 평가 함수 (개선된 버전)
        """
        # 기본: 점수 차이
        score_diff = state.scores[0] - state.scores[1]
        
        # 게임이 끝났으면 큰 보너스/페널티
        if remaining_moves == 0:
            if score_diff > 0:
                return 10000 + score_diff * 100
            elif score_diff < 0:
                return -10000 + score_diff * 100
            else:
                return 0
        
        # 휴리스틱 평가
        eval_score = score_diff * 100
        
        # 남은 움직임이 적을수록 현재 점수가 더 중요
        if remaining_moves < 10:
            eval_score = score_diff * 200
        
        # 추가 휴리스틱들
        moves = board.available_moves()
        
        # 1. 즉시 획득 가능한 박스
        immediate_boxes = sum(board.boxes_completed_by_move(m) for m in moves)
        if state.current_player == 0:
            eval_score += immediate_boxes * 80
        else:
            eval_score -= immediate_boxes * 80
        
        # 2. 안전한 수의 개수
        safe_moves = sum(1 for m in moves if board.danger_score(m) == 0)
        eval_score += safe_moves * 3
        
        # 3. 위험한 박스 수 (2개 선이 그어진 박스)
        dangerous_boxes = 0
        for x in range(state.xsize):
            for y in range(state.ysize):
                edges_count = board.count_edges_of_square((x, y))
                if edges_count == 2:
                    dangerous_boxes += 1
                elif edges_count == 3:
                    # 3개 선이 그어진 박스는 매우 위험
                    dangerous_boxes += 3
        
        eval_score -= dangerous_boxes * 5
        
        return eval_score
    
    def _order_moves(self, board: DotsAndBoxesBoard, moves: List[Move]) -> List[Move]:
        """
        무브 오더링: 더 유망한 수를 먼저 탐색 (Alpha-Beta 효율 증가)
        """
        def move_priority(move: Move) -> Tuple[int, int, int, float]:
            # 1. 박스를 완성하는 수 (가장 우선, 많이 완성할수록 좋음)
            boxes = board.boxes_completed_by_move(move)
            
            # 2. 3개 선이 그어진 박스에 인접한 수는 피하기 (상대에게 기회 제공)
            adjacent_three_edge_boxes = 0
            for square in board.adjacent_squares(move):
                if board.count_edges_of_square(square) == 3:
                    adjacent_three_edge_boxes += 1
            
            # 3. 위험도 (낮을수록 좋음)
            danger = board.danger_score(move)
            
            # 4. 랜덤 타이브레이커
            random_tie = self._rng.random()
            
            # 박스 완성이 최우선, 그 다음은 3개 선 박스 회피, 그 다음은 위험도
            return (-boxes, adjacent_three_edge_boxes, danger, random_tie)
        
        return sorted(moves, key=move_priority)
    
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
    
    def print_stats(self):
        """게임 통계 출력 (디버깅용)"""
        if self.total_moves > 0:
            print(f"\n📊 MinMax 깊이 사용 통계 (총 {self.total_moves}수):")
            for depth in sorted(self.depth_usage.keys()):
                count = self.depth_usage.get(depth, 0)
                percentage = (count / self.total_moves) * 100
                print(f"  Depth {depth}: {count}회 ({percentage:.1f}%)")


# main.py와 호환되는 인터페이스
model: Optional[DynamicDepthMinMaxAgent] = None


def init():
    global model
    model = DynamicDepthMinMaxAgent(time_limit=0.8)


def run(board_lines, xsize, ysize):
    if model is None:
        init()
    move = model.select_move(board_lines, xsize, ysize)
    return [move.x, move.y, move.z]

