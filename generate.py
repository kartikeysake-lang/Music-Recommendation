"""
generate.py
===========
Step 5 of AI Music Generation:
Loads the trained LSTM model and note mappings, generates a novel sequence
of notes using temperature-based sampling, and converts the sequence into
a playable MIDI file saved in the 'output/' directory.
"""

import os
import random
import pickle
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("Error: 'torch' is not installed.")
    print("Please install it by running: pip install torch")
    exit(1)

try:
    from music21 import stream, note, chord, instrument
except ImportError:
    print("Error: 'music21' is not installed.")
    print("Please install it by running: pip install music21")
    exit(1)

# Import our custom LSTM model architecture from train.py
from train import MusicLSTM


DATA_FILE = "preprocessed_data.pkl"
MODEL_FILE = "music_lstm_model.pth"
OUTPUT_DIR = "output"
OUTPUT_MIDI_PATH = os.path.join(OUTPUT_DIR, "generated_music.mid")

# Generation parameters
NUM_NOTES_TO_GENERATE = 100   # Number of notes to create
TEMPERATURE = 0.8            # Controls randomness: 0.5 = conservative/repetitive, 1.0 = creative/varied


def sample_with_temperature(logits, temperature=0.8):
    """
    Applies temperature scaling to the logits and samples a note index.
    A lower temperature produces more predictable and harmonic notes.
    A higher temperature produces more diverse notes.
    """
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    # Sample one index from the categorical probability distribution
    sampled_index = torch.multinomial(probabilities, num_samples=1).item()
    return sampled_index


def generate_notes(model, start_sequence, int_to_note, num_notes=NUM_NOTES_TO_GENERATE, temperature=TEMPERATURE):
    """
    Uses the trained LSTM model in a loop to predict the next note,
    updating the input sequence sliding window at each step.
    """
    model.eval()
    pattern = list(start_sequence)
    predicted_notes = []

    print(f"[+] Generating {num_notes} musical notes (Temperature = {temperature})...")

    with torch.no_grad():
        for step in range(num_notes):
            # Format input as a tensor of shape (1, sequence_length)
            input_tensor = torch.tensor([pattern], dtype=torch.long)

            # Forward pass through the model
            logits, _ = model(input_tensor)

            # Sample next note index
            predicted_index = sample_with_temperature(logits[0], temperature=temperature)
            predicted_note_str = int_to_note[predicted_index]
            predicted_notes.append(predicted_note_str)

            # Slide window: append the predicted note and remove the oldest note
            pattern.append(predicted_index)
            pattern = pattern[1:]

    return predicted_notes


def save_notes_to_midi(predicted_notes, output_path=OUTPUT_MIDI_PATH):
    """
    Converts note and chord strings into music21 Note and Chord objects,
    arranges them in sequence with proper musical timing, and writes a MIDI file.
    """
    offset = 0.0
    output_notes = []

    # Assign Piano instrument to the track
    output_notes.append(instrument.Piano())

    for pattern in predicted_notes:
        # Pattern is a Chord if it contains '.' or is purely numeric
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a single Note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)

        # Increment offset by 0.5 quarter lengths (equivalent to an eighth note rhythm)
        offset += 0.5

    # Create a music21 Stream and write out to standard MIDI format (.mid)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)
    print(f"[+] Successfully wrote MIDI music file to: '{output_path}'")


def main():
    print("=" * 60)
    print("         Step 5: Generate Music with Trained LSTM")
    print("=" * 60)

    # 1. Validate required files exist
    if not os.path.exists(DATA_FILE):
        print(f"[!] Preprocessed data file '{DATA_FILE}' not found.")
        print("    Please run 'python preprocess.py' first.")
        return

    if not os.path.exists(MODEL_FILE):
        print(f"[!] Trained model file '{MODEL_FILE}' not found.")
        print("    Please run 'python train.py' first to train and save the model.")
        return

    # 2. Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. Load note mappings and sequence data
    print(f"[+] Loading dataset mappings from '{DATA_FILE}'...")
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)

    int_to_note = data["int_to_note"]
    network_input = data["network_input"]
    vocab_size = data["vocab_size"]
    sequence_length = data["sequence_length"]

    # 4. Load trained model weights
    print(f"[+] Loading trained model weights from '{MODEL_FILE}'...")
    checkpoint = torch.save if False else torch.load(MODEL_FILE, map_location="cpu", weights_only=False)

    model = MusicLSTM(
        vocab_size=vocab_size,
        embedding_dim=checkpoint.get("embedding_dim", 64),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.2)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 5. Pick a random seed sequence from the training dataset as starter
    seed_index = random.randint(0, len(network_input) - 1)
    seed_sequence = network_input[seed_index]

    # 6. Generate notes
    generated_notes = generate_notes(
        model=model,
        start_sequence=seed_sequence,
        int_to_note=int_to_note,
        num_notes=NUM_NOTES_TO_GENERATE,
        temperature=TEMPERATURE
    )

    # 7. Convert and save to playable MIDI file
    save_notes_to_midi(generated_notes, OUTPUT_MIDI_PATH)

    print("=" * 60)
    print("                 MUSIC GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Generated File: {os.path.abspath(OUTPUT_MIDI_PATH)}")
    print("\nHow to listen to your generated music:")
    print("1. Open with Windows Media Player, VLC, or Groove Music.")
    print("2. Or upload it to an online visualizer like https://midiplayer.ehubsoft.com")
    print("=" * 60)


if __name__ == "__main__":
    main()
