"""
preprocess.py
==============
Step 2 of AI Music Generation:
Reads MIDI files from the 'data/' folder using music21, extracts musical
notes and chords, and converts them into sequences suitable for training an LSTM.
"""

import os
import glob
import pickle
import numpy as np

# music21 is used for reading, parsing, and creating MIDI music files
try:
    from music21 import converter, instrument, note, chord
except ImportError:
    print("Error: 'music21' is not installed.")
    print("Please install it by running: pip install music21")
    exit(1)


DATA_DIR = "data"
OUTPUT_FILE = "preprocessed_data.pkl"
SEQUENCE_LENGTH = 32  # Number of past notes the LSTM will look at to predict the next note


def extract_notes_from_midi(data_folder):
    """
    Parses all MIDI files (.mid, .midi) in the given folder.
    Extracts individual notes and chord groupings into a single list.
    """
    notes = []
    midi_files = glob.glob(os.path.join(data_folder, "*.mid")) + glob.glob(os.path.join(data_folder, "*.midi"))

    if not midi_files:
        print(f"[!] No MIDI files found in folder: '{data_folder}'")
        print("    Please place classical or jazz .mid files into the 'data/' directory.")
        print("    Recommended sources:")
        print("    - Classical MIDI files: http://www.piano-midi.de/")
        print("    - Free Classical Archives: https://www.mfiles.co.uk/classical-midi.htm")
        return notes

    print(f"[+] Found {len(midi_files)} MIDI file(s) in '{data_folder}'. Starting parsing...")

    for file_index, file_path in enumerate(midi_files, 1):
        file_name = os.path.basename(file_path)
        print(f"    [{file_index}/{len(midi_files)}] Parsing: {file_name}")

        try:
            # Parse MIDI file into a music21 score stream
            midi_score = converter.parse(file_path)

            # Extract musical notes and chords from the score
            for element in midi_score.recurse().notes:
                # If the element is a single Note, save its pitch (e.g. 'C4', 'G#5')
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                # If the element is a Chord (multiple notes played together)
                elif isinstance(element, chord.Chord):
                    # Represent chord as dot-separated normal order pitch integers, e.g. '0.4.7'
                    chord_pattern = ".".join(str(n) for n in element.normalOrder)
                    notes.append(chord_pattern)

        except Exception as error:
            print(f"    [Warning] Could not parse '{file_name}': {error}")
            continue

    print(f"[+] Successfully extracted {len(notes)} total notes/chords across all files.")
    return notes


def prepare_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    """
    Maps notes to unique integer IDs and creates sliding window sequences
    (input sequence of length 32 -> target next note).
    """
    if len(notes) <= sequence_length:
        print(f"[!] Not enough notes extracted ({len(notes)}). Need at least {sequence_length + 1} notes.")
        return None, None, None, None

    # Step A: Get all unique note/chord names sorted alphabetically
    unique_notes = sorted(list(set(notes)))
    vocab_size = len(unique_notes)
    print(f"[+] Vocabulary size (unique notes and chords): {vocab_size}")

    # Step B: Create dictionaries to map between note names and integer IDs
    note_to_int = {note_name: index for index, note_name in enumerate(unique_notes)}
    int_to_note = {index: note_name for index, note_name in enumerate(unique_notes)}

    # Step C: Create training input sequences and their corresponding target notes
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        # Sequence of input notes
        sequence_in = notes[i : i + sequence_length]
        # Single target note immediately following the sequence
        target_note = notes[i + sequence_length]

        # Convert strings to integers using the mapping
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[target_note])

    network_input = np.array(network_input, dtype=np.int64)
    network_output = np.array(network_output, dtype=np.int64)

    print(f"[+] Generated {len(network_input)} training sequences (window length = {sequence_length}).")
    return network_input, network_output, note_to_int, int_to_note


def main():
    print("=" * 60)
    print("         Step 2: MIDI Data Preprocessing")
    print("=" * 60)

    # 1. Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[+] Created '{DATA_DIR}/' folder.")

    # 2. Extract notes from MIDI files
    notes = extract_notes_from_midi(DATA_DIR)
    if not notes:
        print("[!] Preprocessing stopped: No MIDI data to process.")
        return

    # 3. Prepare sequences for training
    network_input, network_output, note_to_int, int_to_note = prepare_sequences(notes, SEQUENCE_LENGTH)
    if network_input is None:
        return

    # 4. Save preprocessed dataset and mappings to disk
    dataset_payload = {
        "network_input": network_input,
        "network_output": network_output,
        "note_to_int": note_to_int,
        "int_to_note": int_to_note,
        "vocab_size": len(note_to_int),
        "sequence_length": SEQUENCE_LENGTH,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(dataset_payload, f)

    print(f"[+] Preprocessing complete! Saved processed data to: '{OUTPUT_FILE}'")
    print("=" * 60)
    print("Next step: Run 'python train.py' to train the LSTM model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
