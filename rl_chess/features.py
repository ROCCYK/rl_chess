from __future__ import annotations

from typing import Iterable

import chess
import numpy as np

FEATURE_NAMES = (
    "bias",
    "material_balance",
    "mobility_balance",
    "center_control_balance",
    "king_danger_balance",
    "castling_balance",
    "passed_pawn_balance",
    "development_balance",
    "check_pressure",
    "tempo",
)

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
def _normalize(value: float, scale: float) -> float:
    return float(np.clip(value / scale, -1.0, 1.0))


def _activity_score(board: chess.Board, color: chess.Color) -> int:
    activity = 0
    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue
        activity += len(board.attacks(square))
    return activity


def _material_balance(board: chess.Board) -> float:
    white_score = 0.0
    black_score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        white_score += len(board.pieces(piece_type, chess.WHITE)) * value
        black_score += len(board.pieces(piece_type, chess.BLACK)) * value
    return _normalize(white_score - black_score, 39.0)


def _mobility_balance(board: chess.Board) -> float:
    white_activity = _activity_score(board, chess.WHITE)
    black_activity = _activity_score(board, chess.BLACK)
    return _normalize(float(white_activity - black_activity), 30.0)


def _center_control_balance(board: chess.Board) -> float:
    white_control = 0
    black_control = 0
    for square in CENTER_SQUARES:
        white_control += len(board.attackers(chess.WHITE, square))
        black_control += len(board.attackers(chess.BLACK, square))
    return _normalize(float(white_control - black_control), 8.0)


def _king_danger(board: chess.Board, color: chess.Color) -> float:
    king_square = board.king(color)
    if king_square is None:
        return 8.0

    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square])
    enemy_color = not color
    return float(sum(len(board.attackers(enemy_color, target)) for target in ring))


def _king_danger_balance(board: chess.Board) -> float:
    white_danger = _king_danger(board, chess.WHITE)
    black_danger = _king_danger(board, chess.BLACK)
    return _normalize(black_danger - white_danger, 16.0)


def _castling_balance(board: chess.Board) -> float:
    white_castled = int(board.king(chess.WHITE) in (chess.C1, chess.G1))
    black_castled = int(board.king(chess.BLACK) in (chess.C8, chess.G8))

    white_rights = int(board.has_kingside_castling_rights(chess.WHITE)) + int(
        board.has_queenside_castling_rights(chess.WHITE)
    )
    black_rights = int(board.has_kingside_castling_rights(chess.BLACK)) + int(
        board.has_queenside_castling_rights(chess.BLACK)
    )

    balance = (white_castled - black_castled) + 0.5 * (white_rights - black_rights)
    return _normalize(float(balance), 2.0)


def _is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for enemy_square in enemy_pawns:
        enemy_file = chess.square_file(enemy_square)
        enemy_rank = chess.square_rank(enemy_square)
        if abs(enemy_file - file_index) > 1:
            continue
        if color == chess.WHITE and enemy_rank > rank_index:
            return False
        if color == chess.BLACK and enemy_rank < rank_index:
            return False

    return True


def _count_passed_pawns(board: chess.Board, color: chess.Color) -> int:
    return sum(1 for square in board.pieces(chess.PAWN, color) if _is_passed_pawn(board, square, color))


def _passed_pawn_balance(board: chess.Board) -> float:
    white_passed = _count_passed_pawns(board, chess.WHITE)
    black_passed = _count_passed_pawns(board, chess.BLACK)
    return _normalize(float(white_passed - black_passed), 4.0)


def _developed_minor_pieces(squares: Iterable[chess.Square], starting_squares: Iterable[chess.Square]) -> int:
    starting = set(starting_squares)
    return sum(1 for square in squares if square not in starting)


def _development_balance(board: chess.Board) -> float:
    white_developed = _developed_minor_pieces(board.pieces(chess.KNIGHT, chess.WHITE), (chess.B1, chess.G1))
    white_developed += _developed_minor_pieces(board.pieces(chess.BISHOP, chess.WHITE), (chess.C1, chess.F1))

    black_developed = _developed_minor_pieces(board.pieces(chess.KNIGHT, chess.BLACK), (chess.B8, chess.G8))
    black_developed += _developed_minor_pieces(board.pieces(chess.BISHOP, chess.BLACK), (chess.C8, chess.F8))

    return _normalize(float(white_developed - black_developed), 4.0)


def _check_pressure(board: chess.Board) -> float:
    if not board.is_check():
        return 0.0
    return 1.0 if board.turn == chess.BLACK else -1.0


def extract_features(board: chess.Board) -> np.ndarray:
    return np.array(
        [
            1.0,
            _material_balance(board),
            _mobility_balance(board),
            _center_control_balance(board),
            _king_danger_balance(board),
            _castling_balance(board),
            _passed_pawn_balance(board),
            _development_balance(board),
            _check_pressure(board),
            1.0 if board.turn == chess.WHITE else -1.0,
        ],
        dtype=float,
    )
