"""
Tournament MinMax Agent - 새 규칙 최적화
24초 시간 제한, 20라운드 대회용 최적화
"""

import random
import time
from typing import List, Optional, Sequence, Set, Tuple, Dict
from dataclasses import dataclass

from main import Move, DotsAndBoxesBoard, bitmap_to_edges


@dataclass
class TournamentGameState:
    """대회용 게임 상태"""
    edges: Set[Move]
    scores: Tuple[int, int]
    current_player: int
    xsize: int
    ysize: int
    
    def copy(self):
        return TournamentGameState(
            edges=self.edges.copy(),
            scores=self.scores,
            current_player=self.current_player,
            xsize=self.xsize,
            ysize=self.ysize
        )
    
    def get_hash(self) -> int:
        """빠른 해시 생성"""
        return hash((
            tuple(sorted((m.x, m.y, m.z) for m in self.edges)),
            self.current_player
        ))


class TournamentMinMaxAgent:
    """
    대회용 MinMax 에이전트
    - 24초 시간 제한 최대 활용
    - 20라운드 최고 성능 목표
    - 깊은 탐색으로 정확도 극대화
    """
    
    def __init__(
        self, 
        seed: Optional[int] = None,
        time_limit: float = 2.0  # 빠른 테스트를 위해 2초로 단축
    ):
        self._rng = random.Random(seed)
        self.time_limit = time_limit
        self.nodes_searched = 0
        
        # 대용량 캐시 (24초 동안 최대 활용)
        self.transposition_table: Dict[int, Tuple[int, float, Optional[Move]]] = {}
        
        self.start_time = 0.0
        self.timeout_occurred = False
        
        # 성능 통계
        self.total_moves = 0
        self.cache_hits = 0
        self.max_depth_reached = 0
        
    def _get_tournament_depth(self, available_moves: int, time_remaining: float) -> int:
        """대회용 동적 깊이 - 시간을 최대한 활용"""
        
        # 시간이 충분하면 더 깊게
        if time_remaining > 15.0:
            base_depth = 6
        elif time_remaining > 10.0:
            base_depth = 5
        elif time_remaining > 5.0:
            base_depth = 4
        else:
            base_depth = 3
        
        # 게임 진행도에 따른 조정
        if available_moves > 45:  # 초반
            return max(base_depth - 2, 3)
        elif available_moves > 30:  # 중반
            return max(base_depth - 1, 4)
        elif available_moves > 15:  # 후반
            return base_depth
        elif available_moves > 5:   # 종반
            return min(base_depth + 1, 8)
        else:  # 마지막
            return min(available_moves + 2, 10)  # 거의 완전 탐색
    
    def select_move(self, board_lines: Sequence, xsize: int, ysize: int) -> Move:
        """대회용 최적 수 선택"""
        self.start_time = time.time()
        self.nodes_searched = 0
        self.timeout_occurred = False
        
        edges = bitmap_to_edges(board_lines, xsize, ysize)
        board = DotsAndBoxesBoard(xsize, ysize, edges)
        moves = board.available_moves()
        
        if not moves:
            raise ValueError("더 이상 둘 수 있는 선이 없습니다.")
        
        if len(moves) == 1:
            return moves[0]
        
        # 1단계: 즉시 박스 완성 수 확인
        immediate_wins = []
        for move in moves:
            completed = board.boxes_completed_by_move(move)
            if completed > 0:
                immediate_wins.append((move, completed))
        
        # 여러 완성 수가 있으면 MinMax로 최적 선택
        if len(immediate_wins) > 1:
            return self._select_best_completion_move(immediate_wins, board, xsize, ysize)
        elif len(immediate_wins) == 1:
            return immediate_wins[0][0]
        
        # 2단계: 전체 수에 대해 반복 깊이 탐색 (Iterative Deepening)
        return self._iterative_deepening_search(moves, board, xsize, ysize)
    
    def _select_best_completion_move(self, completion_moves: List[Tuple[Move, int]], 
                                   board: DotsAndBoxesBoard, xsize: int, ysize: int) -> Move:
        """여러 완성 수 중 최적 선택"""
        
        # 완성 개수가 가장 많은 것들만 고려
        max_completion = max(move[1] for move in completion_moves)
        best_moves = [move[0] for move in completion_moves if move[1] == max_completion]
        
        if len(best_moves) == 1:
            return best_moves[0]
        
        # 여러 개면 짧은 시간으로 MinMax 평가
        initial_state = TournamentGameState(
            edges=set(board.edges),
            scores=(0, 0),
            current_player=0,
            xsize=xsize,
            ysize=ysize
        )
        
        best_move = None
        best_score = float('-inf')
        
        for move in best_moves:
            try:
                new_state = self._apply_move(initial_state, move, board)
                score = self._tournament_minimax(
                    new_state, 3, float('-inf'), float('inf'), False, board
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            except TimeoutError:
                break
        
        return best_move or best_moves[0]
    
    def _iterative_deepening_search(self, moves: List[Move], board: DotsAndBoxesBoard, 
                                  xsize: int, ysize: int) -> Move:
        """반복 깊이 탐색 - 시간을 최대한 활용"""
        
        initial_state = TournamentGameState(
            edges=set(board.edges),
            scores=(0, 0),
            current_player=0,
            xsize=xsize,
            ysize=ysize
        )
        
        best_move = moves[0]  # 기본값
        
        # 수 정렬 (좋은 수부터)
        ordered_moves = self._order_moves_advanced(moves, board)
        
        # 깊이를 점진적으로 증가 (빠른 테스트용)
        for depth in range(2, 6):  # 최대 깊이 6까지 (속도 향상)
            
            time_remaining = self.time_limit - (time.time() - self.start_time)
            if time_remaining < 0.5:  # 0.5초 미만이면 중단
                break
            
            try:
                current_best = None
                current_best_score = float('-inf')
                
                for move in ordered_moves:
                    new_state = self._apply_move(initial_state, move, board)
                    
                    score = self._tournament_minimax(
                        new_state, depth - 1, float('-inf'), float('inf'), False, board
                    )
                    
                    if score > current_best_score:
                        current_best_score = score
                        current_best = move
                
                # 이 깊이에서 완료되면 결과 업데이트
                if current_best:
                    best_move = current_best
                    self.max_depth_reached = depth
                
            except TimeoutError:
                self.timeout_occurred = True
                break
        
        self.total_moves += 1
        return best_move
    
    def _apply_move(self, state: TournamentGameState, move: Move, board: DotsAndBoxesBoard) -> TournamentGameState:
        """수 적용"""
        new_edges = state.edges.copy()
        new_edges.add(move)
        
        completed = board.boxes_completed_by_move(move)
        
        new_scores = list(state.scores)
        new_scores[state.current_player] += completed
        
        next_player = state.current_player if completed > 0 else 1 - state.current_player
        
        return TournamentGameState(
            edges=new_edges,
            scores=tuple(new_scores),
            current_player=next_player,
            xsize=state.xsize,
            ysize=state.ysize
        )
    
    def _tournament_minimax(
        self, 
        state: TournamentGameState, 
        depth: int, 
        alpha: float, 
        beta: float, 
        maximizing: bool,
        board: DotsAndBoxesBoard
    ) -> float:
        """대회용 MinMax - 최고 성능"""
        
        self.nodes_searched += 1
        
        # 시간 체크 (덜 빈번하게 - 속도 향상)
        if self.nodes_searched % 10000 == 0:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
        
        # 깊이 0 또는 게임 종료
        if depth == 0:
            return self._tournament_evaluate(state, board)
        
        # 트랜스포지션 테이블 확인
        state_hash = state.get_hash()
        if state_hash in self.transposition_table:
            cached_depth, cached_score, _ = self.transposition_table[state_hash]
            if cached_depth >= depth:
                self.cache_hits += 1
                return cached_score
        
        # 가능한 수들
        temp_board = DotsAndBoxesBoard(state.xsize, state.ysize, state.edges)
        moves = temp_board.available_moves()
        
        if not moves:
            return self._tournament_evaluate(state, board)
        
        # 수 정렬
        moves = self._order_moves_advanced(moves, temp_board)
        
        if maximizing:
            max_eval = float('-inf')
            best_move = None
            
            for move in moves:
                new_state = self._apply_move(state, move, temp_board)
                
                completed = temp_board.boxes_completed_by_move(move)
                next_maximizing = maximizing if completed > 0 else not maximizing
                
                eval_score = self._tournament_minimax(
                    new_state, depth - 1, alpha, beta, next_maximizing, temp_board
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # 알파베타 가지치기
            
            # 캐시 저장
            self.transposition_table[state_hash] = (depth, max_eval, best_move)
            return max_eval
            
        else:
            min_eval = float('inf')
            best_move = None
            
            for move in moves:
                new_state = self._apply_move(state, move, temp_board)
                
                completed = temp_board.boxes_completed_by_move(move)
                next_maximizing = maximizing if completed > 0 else not maximizing
                
                eval_score = self._tournament_minimax(
                    new_state, depth - 1, alpha, beta, next_maximizing, temp_board
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            # 캐시 저장
            self.transposition_table[state_hash] = (depth, min_eval, best_move)
            return min_eval
    
    def _order_moves_advanced(self, moves: List[Move], board: DotsAndBoxesBoard) -> List[Move]:
        """고급 수 정렬"""
        
        def advanced_priority(move: Move) -> Tuple[int, int, int, float]:
            # 1순위: 박스 완성 (높을수록 좋음)
            completed = board.boxes_completed_by_move(move)
            
            # 2순위: 위험도 (낮을수록 좋음)  
            danger = board.danger_score(move)
            
            # 3순위: 중앙 선호 (중앙에 가까울수록 좋음)
            center_distance = abs(move.x - 2.5) + abs(move.y - 2.5)
            
            # 4순위: 랜덤
            random_tie = self._rng.random()
            
            return (-completed, danger, int(center_distance * 10), random_tie)
        
        return sorted(moves, key=advanced_priority)
    
    def _tournament_evaluate(self, state: TournamentGameState, board: DotsAndBoxesBoard) -> float:
        """대회용 정밀 평가 함수"""
        
        # 기본 점수 차이 (가장 중요)
        score_diff = state.scores[0] - state.scores[1]
        
        # 보드 분석
        position_value = 0.0
        
        # 박스 완성도 분석
        three_edge_boxes = 0
        two_edge_boxes = 0
        one_edge_boxes = 0
        
        for x in range(state.xsize):
            for y in range(state.ysize):
                temp_board = DotsAndBoxesBoard(state.xsize, state.ysize, state.edges)
                edges = temp_board.count_edges_of_square((x, y))
                
                if edges == 3:
                    three_edge_boxes += 1
                elif edges == 2:
                    two_edge_boxes += 1
                elif edges == 1:
                    one_edge_boxes += 1
        
        # 가중치 적용
        position_value += three_edge_boxes * 15   # 3변 박스 매우 중요
        position_value += two_edge_boxes * 3      # 2변 박스 중요
        position_value += one_edge_boxes * 0.5    # 1변 박스 약간 중요
        
        # 체인 기회 분석
        chain_potential = self._analyze_chain_potential(state)
        position_value += chain_potential * 8
        
        # 제어권 분석
        control_value = self._analyze_control(state)
        position_value += control_value * 5
        
        # 현재 플레이어에 따라 부호 조정
        if state.current_player == 1:
            position_value = -position_value
        
        return score_diff * 1000 + position_value
    
    def _analyze_chain_potential(self, state: TournamentGameState) -> float:
        """체인 기회 분석"""
        # 간단한 체인 분석 (성능상 제한적)
        temp_board = DotsAndBoxesBoard(state.xsize, state.ysize, state.edges)
        
        chain_count = 0
        for x in range(state.xsize - 1):
            for y in range(state.ysize):
                # 인접한 두 박스가 모두 3변이면 체인 기회
                edges1 = temp_board.count_edges_of_square((x, y))
                edges2 = temp_board.count_edges_of_square((x + 1, y))
                
                if edges1 == 3 and edges2 == 3:
                    chain_count += 1
        
        return chain_count
    
    def _analyze_control(self, state: TournamentGameState) -> float:
        """제어권 분석"""
        total_edges = state.xsize * (state.ysize + 1) + state.ysize * (state.xsize + 1)
        progress = len(state.edges) / total_edges
        
        # 게임 진행도에 따른 제어권 가중치
        if progress < 0.3:
            return 0.3  # 초반은 덜 중요
        elif progress < 0.7:
            return 1.0  # 중반은 중요
        else:
            return 1.5  # 후반은 매우 중요
    
    def get_performance_stats(self) -> Dict:
        """성능 통계"""
        cache_hit_rate = self.cache_hits / max(self.nodes_searched, 1)
        
        return {
            'nodes_searched': self.nodes_searched,
            'cache_size': len(self.transposition_table),
            'cache_hit_rate': cache_hit_rate,
            'max_depth_reached': self.max_depth_reached,
            'total_moves': self.total_moves,
            'timeout_occurred': self.timeout_occurred
        }
