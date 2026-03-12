from __future__ import annotations

import chess
import streamlit as st

from rl_chess import DEFAULT_MODEL_PATH
from rl_chess.agent import TDAgent
from rl_chess.training import TrainingSummary, train_via_self_play

st.set_page_config(page_title="RL Chess", page_icon="♟", layout="wide")


def init_state() -> None:
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "agent" not in st.session_state:
        st.session_state.agent = TDAgent.load_or_create(DEFAULT_MODEL_PATH)
    if "move_history" not in st.session_state:
        st.session_state.move_history = []
    if "selected_square" not in st.session_state:
        st.session_state.selected_square = None
    if "user_color" not in st.session_state:
        st.session_state.user_color = chess.WHITE
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Select a piece to start."


def reset_game(user_color: chess.Color) -> None:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.selected_square = None
    st.session_state.user_color = user_color
    st.session_state.status_message = "New game started."


def build_user_move(board: chess.Board, from_square: chess.Square, to_square: chess.Square) -> chess.Move | None:
    piece = board.piece_at(from_square)
    if piece is None:
        return None

    promotion = None
    if piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
        promotion = chess.QUEEN

    move = chess.Move(from_square, to_square, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None


def square_label(board: chess.Board, square: chess.Square, selected_square: chess.Square | None, legal_targets: set[int]) -> str:
    piece = board.piece_at(square)
    symbol = piece.unicode_symbol() if piece else "·"
    if square == selected_square:
        return f"[{symbol}]"
    if square in legal_targets:
        return f"{symbol}*"
    return symbol


def get_legal_targets(board: chess.Board, selected_square: chess.Square | None) -> set[int]:
    if selected_square is None:
        return set()
    return {move.to_square for move in board.legal_moves if move.from_square == selected_square}


def handle_board_click(square: chess.Square) -> None:
    board = st.session_state.board
    user_color = st.session_state.user_color

    if board.is_game_over(claim_draw=True) or board.turn != user_color:
        st.session_state.selected_square = None
        return

    selected_square = st.session_state.selected_square
    piece = board.piece_at(square)

    if selected_square is None:
        if piece is not None and piece.color == user_color:
            st.session_state.selected_square = square
            st.session_state.status_message = f"Selected {chess.square_name(square)}."
        else:
            st.session_state.status_message = "Select one of your pieces."
        return

    if square == selected_square:
        st.session_state.selected_square = None
        st.session_state.status_message = "Selection cleared."
        return

    if piece is not None and piece.color == user_color:
        st.session_state.selected_square = square
        st.session_state.status_message = f"Selected {chess.square_name(square)}."
        return

    move = build_user_move(board, selected_square, square)
    if move is None:
        st.session_state.status_message = "Illegal move."
        return

    san = board.san(move)
    board.push(move)
    st.session_state.move_history.append(san)
    st.session_state.selected_square = None
    st.session_state.status_message = f"You played {san}."


def maybe_play_agent_move() -> None:
    board = st.session_state.board
    agent_color = not st.session_state.user_color

    if board.is_game_over(claim_draw=True) or board.turn != agent_color:
        return

    move = st.session_state.agent.choose_move(board, training=False)
    san = board.san(move)
    board.push(move)
    st.session_state.move_history.append(san)
    st.session_state.status_message = f"Agent played {san}."


def render_board(board: chess.Board) -> chess.Square | None:
    user_color = st.session_state.user_color
    selected_square = st.session_state.selected_square
    legal_targets = get_legal_targets(board, selected_square)

    ranks = range(7, -1, -1) if user_color == chess.WHITE else range(0, 8)
    files = range(0, 8) if user_color == chess.WHITE else range(7, -1, -1)
    clicked_square = None

    for rank in ranks:
        columns = st.columns([0.45] + [1] * 8)
        columns[0].markdown(f"**{rank + 1}**")
        for index, file_index in enumerate(files, start=1):
            square = chess.square(file_index, rank)
            label = square_label(board, square, selected_square, legal_targets)
            button_type = "primary" if square == selected_square or square in legal_targets else "secondary"
            if columns[index].button(
                label,
                key=f"square_{square}",
                use_container_width=True,
                type=button_type,
            ):
                clicked_square = square

    file_labels = " ".join(chess.FILE_NAMES[file_index] for file_index in files)
    st.caption(f"Files: {file_labels}")
    return clicked_square


def describe_turn(board: chess.Board) -> str:
    return "White to move" if board.turn == chess.WHITE else "Black to move"


def describe_outcome(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "Game in progress"
    if outcome.winner is None:
        return f"Draw by {outcome.termination.name.replace('_', ' ').lower()}"
    winner = "White" if outcome.winner == chess.WHITE else "Black"
    return f"{winner} wins by {outcome.termination.name.replace('_', ' ').lower()}"


def render_training_controls() -> TrainingSummary | None:
    st.sidebar.subheader("Training")
    episodes = st.sidebar.number_input("Self-play episodes", min_value=50, max_value=5000, value=200, step=50)
    summary: TrainingSummary | None = None

    if st.sidebar.button("Train agent"):
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()

        def update_progress(current: int, total: int) -> None:
            progress.progress(int((current / total) * 100))
            status.caption(f"Training {current}/{total}")

        summary = train_via_self_play(
            agent=st.session_state.agent,
            episodes=int(episodes),
            progress_callback=update_progress,
        )
        st.session_state.agent.save(DEFAULT_MODEL_PATH)
        progress.empty()
        status.empty()
        st.session_state.status_message = (
            f"Training complete: {summary.white_wins} white wins, "
            f"{summary.black_wins} black wins, {summary.draws} draws."
        )

    return summary


init_state()
maybe_play_agent_move()

st.title("RL Chess")
st.caption("Play against a compact TD-learning agent. Train it further from the sidebar whenever you want.")

with st.sidebar:
    st.header("Controls")
    color_choice = st.radio(
        "Play as",
        options=("White", "Black"),
        index=0 if st.session_state.user_color == chess.WHITE else 1,
    )
    if st.button("Start new game"):
        reset_game(chess.WHITE if color_choice == "White" else chess.BLACK)
        st.rerun()

training_summary = render_training_controls()

board = st.session_state.board

left_column, right_column = st.columns([1.35, 1])

with left_column:
    st.subheader("Board")
    clicked_square = render_board(board)
    if clicked_square is not None:
        handle_board_click(clicked_square)
        st.rerun()

with right_column:
    st.subheader("Game")
    st.write(describe_turn(board))
    st.write(describe_outcome(board))
    st.write(st.session_state.status_message)

    if st.button("Clear selection", disabled=st.session_state.selected_square is None):
        st.session_state.selected_square = None
        st.session_state.status_message = "Selection cleared."
        st.rerun()

    if st.button("Undo last full turn", disabled=len(board.move_stack) < 2):
        board.pop()
        board.pop()
        if len(st.session_state.move_history) >= 2:
            st.session_state.move_history = st.session_state.move_history[:-2]
        st.session_state.selected_square = None
        st.session_state.status_message = "Last full turn undone."
        st.rerun()

    if training_summary is not None:
        st.write(
            f"Latest training run: {training_summary.episodes} games, "
            f"avg. {training_summary.average_plies:.1f} plies, "
            f"epsilon {training_summary.final_epsilon:.3f}."
        )

    moves = st.session_state.move_history
    if moves:
        paired_moves = []
        for index in range(0, len(moves), 2):
            white_move = moves[index]
            black_move = moves[index + 1] if index + 1 < len(moves) else ""
            paired_moves.append(f"{index // 2 + 1}. {white_move} {black_move}".strip())
        st.text_area("Move list", value="\n".join(paired_moves), height=360)
    else:
        st.write("No moves yet.")
