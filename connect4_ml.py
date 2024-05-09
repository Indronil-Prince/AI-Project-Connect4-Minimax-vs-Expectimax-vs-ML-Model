import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder

ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))

# Load the dataset into a DataFrame
df = pd.read_csv('connect-4_data.csv', header=None)  # Replace 'connect4_dataset.csv' with the actual filename/path

# Extract features (game states) and labels (outcomes)
X_train = df.iloc[:, :-1].values  # All rows, all columns except the last one
X_train = X_train.reshape((-1, 7, 6))
X_train = np.flip(np.transpose(X_train, (0, 2, 1)), axis=1)
y_train = df.iloc[:, -1].values
print(X_train[0])
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the neural network architecture
model = tf.keras.models.Sequential([
    Flatten(input_shape=(6, 7)),  # Flatten 6x7 game board into 1D array
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3)  # Output layer for win, draw, lose
])

# Compile the model
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=20)

# Save the trained model
model.save('connect4_model')

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
        
# Implement a simple Connect Four game interface using the trained model

def drop_piece(board, col, piece):
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                board[row][col] = piece
                return

def print_board(board):
        for row in board:
            print(' '.join(map(str, row)))
        print()

def check_winner(board, piece):
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if board[row][col] == piece and board[row][col+1] == piece and board[row][col+2] == piece and board[row][col+3] == piece:
                    return True

        # Check vertical
        for col in range(7):
            for row in range(3):
                if board[row][col] == piece and board[row+1][col] == piece and board[row+2][col] == piece and board[row+3][col] == piece:
                    return True

        # Check diagonals
        for row in range(3):
            for col in range(4):
                if board[row][col] == piece and board[row+1][col+1] == piece and board[row+2][col+2] == piece and board[row+3][col+3] == piece:
                    return True
        for row in range(3):
            for col in range(3, 7):
                if board[row][col] == piece and board[row+1][col-1] == piece and board[row+2][col-2] == piece and board[row+3][col-3] == piece:
                    return True

        return False

def is_valid_location(board, col):
        return board[0][col] == 0
        
def evaluate_board(board):
        best_move = None
        all = []
        input_board = np.array(board).reshape((1, 6, 7))
        best_outcome = model.predict(input_board)
        best_outcome = np.argmax(best_outcome)
        
        for col in range(7):
            if is_valid_location(board, col):
                # Copy the board to simulate the move
                temp_board = board.copy()
                drop_piece(temp_board, col, 2)
                
                # Convert the board into a suitable format for the model
                input_board = np.array(temp_board).reshape((1, 6, 7))
                #print(input_board)
                # Predict the outcome using the trained model
                outcome = model.predict(input_board)
                outcome = outcome[0][0]
                print(col, " ", all, "\n")
                all.append(outcome)
            else:
                all.append(-1000000000000000000)

                # Update best move if this move has a better outcome
        best_move = all.index(max(all))
                    
        return best_move

if __name__ == "__main__":
    # Initialize the game board
    board = create_board()

    # Main game loop
    while True:
        print_board(board)
        player_col = int(input("Player's turn (0-6): "))
        drop_piece(board, player_col, 1)
        if check_winner(board, 1):
            print("Player wins!")
            break
        if 0 not in board[0]:
            print("It's a draw!")
            break
        print_board(board)
        # AI's turn
        print("AI's turn:")
        ai_col = evaluate_board(board, model)
        drop_piece(board, ai_col, 2)
        if check_winner(board, 2):
            print("AI wins!")
            break
        if 0 not in board[0]:
            print("It's a draw!")
            break
