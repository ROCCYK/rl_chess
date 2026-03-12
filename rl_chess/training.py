from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import chess

from rl_chess.agent import TDAgent


ProgressCallback = Callable[[int, int], None]


@dataclass
class TrainingSummary:
    episodes: int
    white_wins: int
    black_wins: int
    draws: int
    average_plies: float
    final_epsilon: float


def train_via_self_play(
    agent: TDAgent,
    episodes: int,
    max_plies: int = 180,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    progress_callback: ProgressCallback | None = None,
) -> TrainingSummary:
    white_wins = 0
    black_wins = 0
    draws = 0
    total_plies = 0

    for episode in range(episodes):
        board = chess.Board()
        plies = 0

        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            board_before = board.copy(stack=False)
            move = agent.choose_move(board, training=True)
            board.push(move)
            agent.td_update(board_before, board)
            plies += 1

        if not board.is_game_over(claim_draw=True):
            agent.td_update(board.copy(stack=False), board.copy(stack=False), forced_target=0.0)

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
        elif outcome.winner == chess.WHITE:
            white_wins += 1
        else:
            black_wins += 1

        total_plies += plies
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        if progress_callback is not None:
            progress_callback(episode + 1, episodes)

    average_plies = total_plies / episodes if episodes else 0.0
    return TrainingSummary(
        episodes=episodes,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        average_plies=average_plies,
        final_epsilon=agent.epsilon,
    )
