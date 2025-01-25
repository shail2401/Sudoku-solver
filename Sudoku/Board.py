from typing import Union
import numpy as np
from numpy import array
import numpy

class Board:
    
    _board = np.zeros((9, 9))

    def __init__(self, grid: array) -> None:
        if grid is not None:
            self._board = grid
        self._solved = False

    def __len__(self):
        return len(self._board)

    def __str__(self) -> str:
        res = ''
        for i in range(len(self._board)):
            if i % 3 == 0 and i != 0:
                res += "- - - - - - - - - - - - - \n"
            for j in range(len(self._board[0])):
                if j % 3 == 0 and j != 0:
                    res += " | "
                if j == 8:
                    res += f"{self._board[i][j]}\n"
                else:
                    res += f"{self._board[i][j]} "
        return res

    def __getitem__(self, index):
        return self._board[index]

    def valid(self, num: int, pos: tuple) -> bool:
        for i in range(len(self._board[0])):
            if self._board[pos[0]][i] == num and pos[1] != i:
                return False

        for i in range(len(self._board)):
            if self._board[i][pos[1]] == num and pos[0] != i:
                return False

        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if self._board[i][j] == num and pos != (i, j):
                    return False

        return True

    def find_empty(self) -> Union[tuple, None]:
        for i in range(len(self._board)):
            for j in range(len(self._board[0])):
                if self._board[i][j] == 0:
                    return i, j
        return None
