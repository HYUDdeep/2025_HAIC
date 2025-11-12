from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
# 모델 import
from main import DotsAndBoxesBoard, HeuristicAgent, Move, bitmap_to_edges

def total_edge_count(xsize: int, ysize: int) -> int:
    return xsize * (ysize + 1) + ysize * (xsize + 1)


class DotsAndBoxesGame:
    def __init__(self, xsize: int, ysize: int):
        self.xsize = xsize
        self.ysize = ysize
        self.total_edges = total_edge_count(xsize, ysize)
        self.reset()

    def reset(self):
        self.board_lines = [
            [[0, 0] for _ in range(self.ysize + 1)]
            for _ in range(self.xsize + 1)
        ]
        self.edges: Set[Move] = set()
        self.scores = [0, 0]
        self.current_player = 0
        self.finished = False
        self.winner: Optional[int] = None
        self.illegal_move = False
        self.move_history: List[int] = []

    def board_snapshot(self) -> List[List[List[int]]]:
        return [[cell[:] for cell in column] for column in self.board_lines]

    def is_legal(self, move: Move) -> bool:
        if move.z == 0:
            if not (0 <= move.x < self.xsize and 0 <= move.y <= self.ysize):
                return False
        else:
            if not (0 <= move.x <= self.xsize and 0 <= move.y < self.ysize):
                return False
        return self.board_lines[move.x][move.y][move.z] == 0

    def apply_move(self, move: Move) -> bool:
        if self.finished:
            return False
        if not self.is_legal(move):
            self.finished = True
            self.winner = 1 - self.current_player
            self.illegal_move = True
            return False

        board = DotsAndBoxesBoard(self.xsize, self.ysize, self.edges)
        completed = board.boxes_completed_by_move(move)

        self.board_lines[move.x][move.y][move.z] = 1
        self.edges.add(move)
        self.move_history.extend([move.x, move.y, move.z])

        if completed:
            self.scores[self.current_player] += completed
        else:
            self.current_player = 1 - self.current_player

        if len(self.edges) == self.total_edges:
            self.finished = True
            if self.scores[0] > self.scores[1]:
                self.winner = 0
            elif self.scores[1] > self.scores[0]:
                self.winner = 1
            else:
                self.winner = None

        return True


def evaluate_agents(agent_a, agent_b, games: int, xsize: int, ysize: int) -> dict:
    stats = {
        "games": 0,
        "wins_a": 0,
        "wins_b": 0,
        "illegal_losses_a": 0,
        "illegal_losses_b": 0,
        "avg_score_diff": 0.0,
    }

    for game_idx in range(games):
        if game_idx % 2 == 0:
            result = play_single_game(agent_a, agent_b, xsize, ysize)
            first_is_a = True
        else:
            result = play_single_game(agent_b, agent_a, xsize, ysize)
            first_is_a = False

        score_first, score_second = result["scores"]
        if first_is_a:
            score_a, score_b = score_first, score_second
            winner = result["winner"]
            illegal = result["illegal_move"] and winner is not None
            if winner == 0:
                stats["wins_a"] += 1
                if illegal:
                    stats["illegal_losses_b"] += 1
            elif winner == 1:
                stats["wins_b"] += 1
                if illegal:
                    stats["illegal_losses_a"] += 1
        else:
            score_a, score_b = score_second, score_first
            winner = result["winner"]
            illegal = result["illegal_move"] and winner is not None
            if winner == 0:
                stats["wins_b"] += 1
                if illegal:
                    stats["illegal_losses_a"] += 1
            elif winner == 1:
                stats["wins_a"] += 1
                if illegal:
                    stats["illegal_losses_b"] += 1

        stats["games"] += 1
        stats["avg_score_diff"] += score_a - score_b

    if stats["games"]:
        stats["avg_score_diff"] /= stats["games"]

    return stats


def play_single_game(agent_first, agent_second, xsize: int, ysize: int) -> dict:
    game = DotsAndBoxesGame(xsize, ysize)
    agents = [agent_first, agent_second]

    while not game.finished:
        player = game.current_player
        agent = agents[player]
        board_lines = game.board_snapshot()
        move = agent.select_move(board_lines, xsize, ysize)
        if not isinstance(move, Move):
            move = Move(*move)
        legal = game.apply_move(move)
        if not legal:
            break

    return {
        "scores": tuple(game.scores),
        "winner": game.winner,
        "illegal_move": game.illegal_move,
    }


if __name__ == "__main__":
    heuristic_a = HeuristicAgent(seed=0)
    heuristic_b = HeuristicAgent(seed=1)

    games = 1000
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = Path(__file__).with_name(f"simulation_report_{timestamp}.txt")

    sections: List[str] = []

    results = evaluate_agents(
        # 모델 바꾸기
        heuristic_a,
        heuristic_b,
        games=games,
        xsize=5,
        ysize=5,
    )

    section = ["=== Heuristic A vs Heuristic B ==="]
    section.append(f"총 게임 수: {results['games']}")
    section.append(f"A(Heuristic A) 승리: {results['wins_a']}")
    section.append(f"B(Heuristic B) 승리: {results['wins_b']}")
    section.append(f"평균 점수 차 (A-B): {results['avg_score_diff']:.3f}")
    section.append(
        f"불법 수로 진 A/B: {results['illegal_losses_a']} / {results['illegal_losses_b']}"
    )

    sections.append("\n".join(section))

    report_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
