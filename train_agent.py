from __future__ import annotations

import argparse
from pathlib import Path

from rl_chess import DEFAULT_MODEL_PATH
from rl_chess.agent import TDAgent
from rl_chess.training import train_via_self_play


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the TD-learning chess agent through self-play.")
    parser.add_argument("--episodes", type=int, default=400, help="Number of self-play games to run.")
    parser.add_argument("--max-plies", type=int, default=180, help="Maximum plies per self-play game.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the model.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for training.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="TD learning rate.")
    parser.add_argument("--discount", type=float, default=0.98, help="Bootstrap discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.18, help="Exploration rate for self-play.")
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing model and start from scratch.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.output.exists() and not args.fresh:
        agent = TDAgent.load(args.output)
    else:
        agent = TDAgent()

    agent.learning_rate = args.learning_rate
    agent.discount = args.discount
    agent.epsilon = args.epsilon
    agent.reseed(args.seed)

    summary = train_via_self_play(
        agent=agent,
        episodes=args.episodes,
        max_plies=args.max_plies,
    )
    agent.save(args.output)

    print(f"Saved model to {args.output}")
    print(f"Episodes: {summary.episodes}")
    print(f"White wins: {summary.white_wins}")
    print(f"Black wins: {summary.black_wins}")
    print(f"Draws: {summary.draws}")
    print(f"Average plies: {summary.average_plies:.1f}")
    print(f"Final epsilon: {summary.final_epsilon:.3f}")


if __name__ == "__main__":
    main()
