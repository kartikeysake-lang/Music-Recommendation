# Music-Recommendation
# 🎹 AI Music Recommendation And Generation using LSTM Neural Networks

A beginner-friendly deep learning project that learns classical piano melodies from MIDI files and generates original musical sequences using an **LSTM (Long Short-Term Memory)** Recurrent Neural Network.

---

## 📁 Project Structure

```text
music-recomm/
│
├── data/                    # Put your classical MIDI files (.mid) here
├── output/                  # Generated MIDI files will be saved here
│
├── preprocess.py            # Reads MIDI files using music21 and creates training sequences
├── train.py                 # Defines the LSTM architecture and trains the model
├── generate.py              # Generates novel music from the trained model and saves as MIDI
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation and guide
