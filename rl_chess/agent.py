from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from rl_chess import DEFAULT_MODEL_PATH
from rl_chess.features import FEATURE_NAMES, extract_features


def default_weights() -> np.ndarray:
    return np.array(
        [
            0.0,
            1.15,
            0.18,
            0.12,
            0.16,
            0.08,
            0.12,
            0.10,
            0.25,
            0.02,
        ],
        dtype=float,
    )


def terminal_reward(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0


@dataclass
class TDAgent:
    weights: np.ndarray = field(default_factory=default_weights)
    learning_rate: float = 0.08
    discount: float = 0.98
    epsilon: float = 0.18
    seed: int = 7
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} weights, got {self.weights.shape}.")
        self.rng = np.random.default_rng(self.seed)

    def reseed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def evaluate(self, board: chess.Board) -> float:
        if board.is_game_over(claim_draw=True):
            return terminal_reward(board)
        return math.tanh(float(self.weights @ extract_features(board)))

    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        board.push(move)
        score = self.evaluate(board)
        board.pop()
        return score

    def choose_move(self, board: chess.Board, training: bool = False) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        if training and self.epsilon > 0 and self.rng.random() < self.epsilon:
            return legal_moves[int(self.rng.integers(len(legal_moves)))]

        maximizing = board.turn == chess.WHITE
        best_score: float | None = None
        best_moves: list[chess.Move] = []

        for move in legal_moves:
            score = self.score_move(board, move)
            if best_score is None:
                best_score = score
                best_moves = [move]
                continue

            if maximizing and score > best_score + 1e-9:
                best_score = score
                best_moves = [move]
            elif not maximizing and score < best_score - 1e-9:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(move)

        return best_moves[int(self.rng.integers(len(best_moves)))]

    def td_update(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        forced_target: float | None = None,
    ) -> float:
        features = extract_features(board_before)
        prediction = math.tanh(float(self.weights @ features))

        if forced_target is not None:
            target = forced_target
        elif board_after.is_game_over(claim_draw=True):
            target = terminal_reward(board_after)
        else:
            target = self.discount * self.evaluate(board_after)

        td_error = target - prediction
        gradient = (1.0 - prediction**2) * features
        self.weights += self.learning_rate * td_error * gradient
        self.weights = np.clip(self.weights, -3.0, 3.0)
        return float(td_error)

    def to_payload(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": [float(weight) for weight in self.weights],
            "learning_rate": float(self.learning_rate),
            "discount": float(self.discount),
            "epsilon": float(self.epsilon),
            "seed": int(self.seed),
        }

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "TDAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = payload.get("feature_names", [])
        if list(feature_names) != list(FEATURE_NAMES):
            raise ValueError("Saved model features do not match the current feature extractor.")

        return cls(
            weights=np.array(payload["weights"], dtype=float),
            learning_rate=float(payload.get("learning_rate", 0.08)),
            discount=float(payload.get("discount", 0.98)),
            epsilon=float(payload.get("epsilon", 0.18)),
            seed=int(payload.get("seed", 7)),
        )

    @classmethod
    def load_or_create(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "TDAgent":
        model_path = Path(path)
        if model_path.exists():
            return cls.load(model_path)

        agent = cls()
        agent.save(model_path)
        return agent
