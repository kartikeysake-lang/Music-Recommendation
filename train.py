"""
train.py
========
Steps 3 & 4 of AI Music Generation:
3. Builds an LSTM-based Recurrent Neural Network (RNN) using PyTorch.
4. Trains the model on preprocessed MIDI sequences and saves the trained model weights.
"""

import os
import pickle
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    print("Error: 'torch' is not installed.")
    print("Please install it by running: pip install torch")
    exit(1)


DATA_FILE = "preprocessed_data.pkl"
MODEL_SAVE_PATH = "music_lstm_model.pth"

# Training hyperparameters (kept beginner-friendly and fast for CPU training)
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.002
EPOCHS = 20


# ==============================================================================
# Step 3: Define the LSTM Architecture
# ==============================================================================
class MusicLSTM(nn.Module):
    """
    LSTM-based Recurrent Neural Network for Music Sequence Modeling.

    Architecture:
    1. Embedding Layer: Maps discrete note IDs to dense vector representations.
    2. 2-layer Stacked LSTM: Captures short-term and long-term musical patterns.
    3. Dropout Layer: Helps prevent overfitting to training melodies.
    4. Fully Connected (Linear) Layer: Projects to the vocabulary of unique notes.
    """
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE):
        super(MusicLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding converts note integer tokens into continuous embedding vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Final projection to output vocabulary size (unnormalized logits)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_sequence, hidden_state=None):
        # input_sequence: (batch_size, sequence_length)
        embedded = self.embedding(input_sequence)
        # embedded: (batch_size, sequence_length, embedding_dim)

        lstm_out, hidden_state = self.lstm(embedded, hidden_state)
        # lstm_out: (batch_size, sequence_length, hidden_dim)

        # Extract the hidden representation of the last step in the sequence
        last_step_out = lstm_out[:, -1, :]
        last_step_out = self.dropout(last_step_out)

        # Output logits for each note in vocabulary
        logits = self.fc(last_step_out)
        return logits, hidden_state


# ==============================================================================
# Step 4: Train the LSTM Model
# ==============================================================================
def train_model():
    print("=" * 60)
    print("         Steps 3 & 4: LSTM Model Training")
    print("=" * 60)

    # 1. Check if preprocessed data exists
    if not os.path.exists(DATA_FILE):
        print(f"[!] Preprocessed data file '{DATA_FILE}' not found.")
        print("    Please run 'python preprocess.py' first to prepare the dataset.")
        return

    print(f"[+] Loading preprocessed dataset from '{DATA_FILE}'...")
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)

    network_input = data["network_input"]
    network_output = data["network_output"]
    vocab_size = data["vocab_size"]
    sequence_length = data["sequence_length"]

    print(f"[+] Dataset Summary:")
    print(f"    - Total training samples : {len(network_input)}")
    print(f"    - Vocabulary size        : {vocab_size} unique notes/chords")
    print(f"    - Input sequence length  : {sequence_length} notes")

    # 2. Select device (CUDA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using compute device     : {device}")

    # 3. Create PyTorch DataLoader
    tensor_x = torch.tensor(network_input, dtype=torch.long)
    tensor_y = torch.tensor(network_output, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Instantiate Model, Loss Function, and Optimizer
    model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[+] Initialized 2-layer LSTM with {sum(p.numel() for p in model.parameters()):,} parameters.")
    print("-" * 60)
    print(f"Starting training for {EPOCHS} epochs (Batch size: {BATCH_SIZE})...")
    print("-" * 60)

    # 5. Training Loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Zero previous gradients
            optimizer.zero_grad()

            # Forward pass
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

            # Calculate batch accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch_y).sum().item()
            total_samples += batch_x.size(0)

        epoch_loss = running_loss / total_samples
        epoch_accuracy = (correct_predictions / total_samples) * 100.0

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}]  |  Loss: {epoch_loss:.4f}  |  Accuracy: {epoch_accuracy:.2f}%")

    # 6. Save the trained model checkpoint
    print("-" * 60)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT_RATE,
        "sequence_length": sequence_length
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"[+] Model successfully trained and saved to: '{MODEL_SAVE_PATH}'")
    print("=" * 60)
    print("Next step: Run 'python generate.py' to generate new music.")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
