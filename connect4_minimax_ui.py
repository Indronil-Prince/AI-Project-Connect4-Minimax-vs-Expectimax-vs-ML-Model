import tkinter as tk
from tkinter import messagebox, font
import numpy as np
# Import the backend code
from connect4_minimax import create_board, drop_piece, is_valid_location, get_next_open_row, winning_move, get_ai_move

class ConnectFourApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Connect Four - Minimax")
        self.board = create_board()
        self.current_player = 1
        self.game_over = False

        self.create_widgets()

    def create_widgets(self):
        self.cells = []
        for row in range(6):
            row_cells = []
            for col in range(7):
                cell = tk.Button(self.master, text="", width=12, height=6, borderwidth=3,
                                 command=lambda row=row, col=col: self.drop_piece(col))
                cell.grid(row=row, column=col)
                row_cells.append(cell)
            self.cells.append(row_cells)

        self.restart_button = tk.Button(self.master, text="Restart Game", command=self.restart_game, padx=10, pady=10, bg="black", fg="white")
        self.restart_button.grid(row=6, columnspan=7)

    def drop_piece(self, col):
        if self.game_over:
            return
        if is_valid_location(self.board, col):
            row = get_next_open_row(self.board, col)
            drop_piece(self.board, row, col, self.current_player)
            self.update_board_ui()
            if winning_move(self.board, self.current_player):
                if self.current_player == 1:
                    messagebox.showinfo("Game Over", f"You win!")
                elif self.current_player == 2:
                    messagebox.showinfo("Game Over", f"AI wins!")
                self.game_over = True
            else:
                self.current_player = 2 if self.current_player == 1 else 1
                if self.current_player == 2:
                    self.ai_move()

    def ai_move(self):
        col = get_ai_move(self.board)
        if is_valid_location(self.board, col):
            row = get_next_open_row(self.board, col)
            drop_piece(self.board, row, col, self.current_player)
            self.update_board_ui()
            if winning_move(self.board, self.current_player):
                if self.current_player == 1:
                    messagebox.showinfo("Game Over", f"You win!")
                elif self.current_player == 2:
                    messagebox.showinfo("Game Over", f"AI wins!")
                self.game_over = True
            else:
                self.current_player = 2 if self.current_player == 1 else 1

    def update_board_ui(self):
        rows = {0:5, 1:4, 2:3, 3:2, 4:1, 5:0}
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == 0:
                    self.cells[rows[row]][col].config(bg="white", text="")
                elif self.board[row][col] == 1:
                    self.cells[rows[row]][col].config(bg="blue", text="Human", fg="white")
                else:
                    self.cells[rows[row]][col].config(bg="red", text="AI", fg="white")

    def restart_game(self):
        self.board = create_board()
        self.current_player = 1
        self.game_over = False
        self.update_board_ui()


def main():
    root = tk.Tk()
    app = ConnectFourApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
