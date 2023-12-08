import librosa
import librosa.feature
import wx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk

# Function to recognize instruments based on spectral features
def recognize_instruments(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    instruments = []
    # Conditions for Piano
    if max(chroma_stft.mean(axis=1)) > 0.3 and max(spec_bw.mean(axis=1)) < 2500:
        instruments.append("Piano")
    # Conditions for Violin
    if max(chroma_stft.mean(axis=1)) > 0.25 and max(spec_cent.mean(axis=1)) > 3000:
        instruments.append("Violin")
    # Conditions for Cello
    if max(rolloff.mean(axis=1)) < 3500 and max(spec_cent.mean(axis=1)) < 2500:
        instruments.append("Cello")
    # Conditions for Flute
    if max(spec_cent.mean(axis=1)) > 3500 and max(zcr.mean(axis=1)) < 0.1:
        instruments.append("Flute")
    # Conditions for Oboe
    if max(chroma_stft.mean(axis=1)) > 0.35 and max(spec_cent.mean(axis=1)) > 2500:
        instruments.append("Oboe")

    return instruments

# Function to select a file
def select_file():
    app = wx.App()

    # Select a local mp3 file
    with wx.FileDialog(None, "Choose a File", wildcard="Audio files (*.mp3)|*.mp3",
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
        if file_dialog.ShowModal() == wx.ID_CANCEL:
            raise Exception("No File Selected")

        file_path = file_dialog.GetPath()
        print("File Path:", file_path)
        return file_path

# Function to extract MFCC features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfccs.T, axis=0)

# Dataset
data = [
    ('./mp3data/celloSolo1.mp3', 'cello'),
    ('./mp3data/celloSolo2.mp3', 'cello'),
    ('./mp3data/fluteSolo1.mp3', 'flute'),
    ('./mp3data/fluteSolo2.mp3', 'flute'),
    ('./mp3data/harpSolo1.mp3', 'harp'),
    ('./mp3data/harpSolo2.mp3', 'harp'),
    ('./mp3data/oboeSolo1.mp3', 'oboe'),
    ('./mp3data/pianoSolo1.mp3', 'piano'),
    ('./mp3data/violinSolo1.mp3', 'violin'),
    # ......
]

# Prepare the data
X, y = [], []
for audio_path, label in data:
    features = extract_features(audio_path)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function for ML-based prediction
def predict_instrument(audio_path):
    features = extract_features(audio_path)
    prediction = model.predict([features])
    print(f'Prediction: {prediction}')
    return prediction[0]

# Combined prediction function
def combined_predict_instrument(audio_path):
    # ML-based prediction
    ml_prediction = predict_instrument(audio_path)

    # Spectral feature-based prediction
    spectral_prediction = recognize_instruments(audio_path)

    return ml_prediction, spectral_prediction

# Function to show a message popup
def popup1(txt):
    # Create the main window
    root = tk.Tk()
    root.geometry("400x250")
    root.title("Instrument Prediction")
    # Create a label widget
    label = tk.Label(root, text=txt)
    label.pack(pady=100)
    # Start the Tkinter event loop
    root.mainloop()

# Main program
if __name__ == '__main__':
    audio_file = select_file()

    # Combined prediction
    ml_instrument, spectral_instruments = combined_predict_instrument(audio_file)

    message = f"ML Prediction: {ml_instrument}\nSpectral Feature Prediction: {', '.join(spectral_instruments)}"
    popup1(message)