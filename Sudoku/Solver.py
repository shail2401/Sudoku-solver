from . import Board


class Solver:

    def __init__(self, board: Board):
        self._board = Board.Board(board)
        self.solve()

    def solve(self):
        find = self._board.find_empty()
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if self._board.valid(i, (row, col)):
                self._board[row][col] = i

                if self.solve():
                    return True

                self._board[row][col] = 0
        return False
