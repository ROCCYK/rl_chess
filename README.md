# RL Chess

A Streamlit chess app where a human can play against a temporal-difference (TD) learning agent.

The agent uses a small linear value function over chess features such as material, mobility, center control, king danger, passed pawns, and development. It is intentionally lightweight so the app stays easy to run locally and can keep training through self-play.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
python train_agent.py --episodes 400
streamlit run app.py
```

## What is included

- `app.py`: Streamlit interface for human vs agent games.
- `train_agent.py`: CLI entrypoint for self-play training.
- `rl_chess/features.py`: Hand-crafted feature extractor.
- `rl_chess/agent.py`: TD-learning agent with JSON persistence.
- `rl_chess/training.py`: Self-play training loop and summary metrics.

## Notes

- If `models/td_agent.json` does not exist yet, the app will create a warm-start model automatically.
- User moves are made by clicking a piece and then a destination square.
- Promotions default to a queen to keep the UI simple.
- The model is not engine-grade; it is a compact RL demo designed for interactive play and continued training.
