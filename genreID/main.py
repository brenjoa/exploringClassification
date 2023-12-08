import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import tkinter as tk
from tkinter import filedialog


# Extract MFCC features
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, mono=True, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {audio_path}")
        return None
    return mfccs_processed


# Read excel file and extract features
def extract_features_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    features, labels = [], []

    for _, row in df.iterrows():
        file_path = row['audio_path']
        genre = row['genre']
        mfccs = extract_features('./mp3data/' + file_path)

        if mfccs is not None:
            features.append(mfccs)
            labels.append(genre)

    return np.array(features), np.array(labels)


# Train and evaluate an SVM model
def train_model(features, labels):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Make a prediction
    y_pred = model.predict(X_test)

    # Print results
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')

    return model, scaler


# Predit the genre
def predict_genre(model, scaler):
    # Select an audio file
    audio_path = filedialog.askopenfilename()

    # Extract features
    features = extract_features(audio_path)

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict the genre
    prediction = model.predict(features_scaled)

    # Update the result label
    result_label.config(text=f"Predicted Genre: {prediction[0]}")


# Create GUI
def create_gui(model, scaler):
    root = tk.Tk()
    root.title("Music Genre Prediction")

    # Set the GUI layout
    select_button = tk.Button(root, text="Select MP3 File", command=lambda: predict_genre(model, scaler))
    select_button.pack()

    global result_label
    result_label = tk.Label(root, text="Predicted Genre: None", fg="blue")
    result_label.pack()

    # Run GUI
    root.mainloop()


# Main
if __name__ == "__main__":
    # Extract features and labels from Excel
    features, labels = extract_features_from_excel('./music.xlsx')

    # Train the model
    model, scaler = train_model(features, labels)

    # Save model and scaler to file
    dump(model, 'test_model.joblib')
    dump(scaler, 'test_scaler.joblib')

    # Load model and scaler
    model_loaded = load('test_model.joblib')
    scaler_loaded = load('test_scaler.joblib')

    # Create GUI
    create_gui(model_loaded, scaler_loaded)