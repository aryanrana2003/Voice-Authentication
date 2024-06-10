import sounddevice as sd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import pickle
import os
import datetime
from twilio.rest import Client

# Global configurations
DATA_DIR = "voice_samples"
MODEL_PATH = "voice_auth_model.pkl"
LOG_FILE = "authentication_log.txt"
DURATION = 5
FS = 44100
THRESHOLD = 0.6  # Probability threshold for authentication

# Twilio configurations
TWILIO_ACCOUNT_SID = 'xyz'    #twilio sid
TWILIO_AUTH_TOKEN = 'xyz'     #authentication token
TWILIO_PHONE_NUMBER = 'xyz'    #twilio phone number
RECIPIENT_PHONE_NUMBER = 'xyz'  # your phone number

def record_audio(duration=DURATION, fs=FS):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete")
    return np.squeeze(audio)

def extract_features(audio, sr=FS):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(contrast.T, axis=0)
    ])
    return features

def augment_audio(audio, sr=FS):
    # Pitch shifting
    audio_pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    # Time stretching
    audio_time_stretched = librosa.effects.time_stretch(audio, rate=1.2)
    # Adding noise
    noise = np.random.randn(len(audio))
    audio_noisy = audio + 0.005 * noise

    return [audio, audio_pitch_shifted, audio_time_stretched, audio_noisy]

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    user_id = input("Enter user ID: ")
    user_dir = os.path.join(DATA_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    existing_samples = len([name for name in os.listdir(user_dir) if name.endswith(".npy")])
    for i in range(existing_samples, existing_samples + 5):  # Collect 5 additional samples per user
        print(f"Recording sample {i + 1} for user {user_id}")
        audio = record_audio()
        augmented_audios = augment_audio(audio)
        for j, aug_audio in enumerate(augmented_audios):
            features = extract_features(aug_audio)
            np.save(os.path.join(user_dir, f"sample_{i + 1}_{j}.npy"), features)
    print(f"Data collection for user {user_id} complete")

def train_model():
    features_list = []
    labels_list = []

    for user_id in os.listdir(DATA_DIR):
        user_dir = os.path.join(DATA_DIR, user_id)
        if os.path.isdir(user_dir):
            print(f"Processing data for user: {user_id}")
            for sample_file in os.listdir(user_dir):
                if sample_file.endswith(".npy"):
                    print(f"Loading sample file: {sample_file}")
                    features = np.load(os.path.join(user_dir, sample_file))
                    features_list.append(features)
                    labels_list.append(user_id)

    unique_classes = set(labels_list)
    print("Unique classes in the dataset:", unique_classes)

    if len(unique_classes) < 2:
        print("Error: At least two classes are required for training.")
        return

    if features_list and labels_list:
        X_train, X_test, y_train, y_test = train_test_split(features_list, labels_list, test_size=0.2, random_state=42)

        # Ensure balanced classes
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Training set class distribution: {dict(zip(unique, counts))}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True))
        ])

        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': [1e-3, 1e-4, 'scale']
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        accuracy = best_model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")

        # Train an Isolation Forest on the same dataset
        isolation_forest = Pipeline([
            ('scaler', StandardScaler()),
            ('iforest', IsolationForest(contamination='auto', random_state=42))
        ])
        isolation_forest.fit(X_train)

        # Save both models
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump({'classifier': best_model, 'anomaly_detector': isolation_forest}, model_file)
    else:
        print("No data available to train the model.")

def send_twilio_notification():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message_body = "Unauthorized access detected in the voice authentication system. Potential Security breach."
    message = client.messages.create(
        body=message_body,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print("Twilio notification sent successfully.")

def authenticate():
    if not os.path.exists(MODEL_PATH):
        print("Model not trained yet. Train the model first.")
        return

    with open(MODEL_PATH, 'rb') as model_file:
        models = pickle.load(model_file)
        classifier = models['classifier']
        anomaly_detector = models['anomaly_detector']

    new_audio = record_audio()
    new_features = extract_features(new_audio)
    new_features_scaled = classifier.named_steps['scaler'].transform([new_features])

    # Check if the audio matches any of the authenticated users
    with open(LOG_FILE, 'r') as log_file:
        for line in log_file:
            parts = line.strip().split(" - ")
            if len(parts) >= 4:
                _, _, status, user_id = parts
                if status == "Authenticated":
                    saved_user_features = np.load(os.path.join(DATA_DIR, user_id, f"sample_1_0.npy"))  # Assuming each user has only one sample for simplicity
                    saved_user_features_scaled = classifier.named_steps['scaler'].transform([saved_user_features])
                    similarity_score = classifier.named_steps['svm'].decision_function([new_features_scaled[0]])
                    if similarity_score > THRESHOLD:
                        print(f"Authenticated user: {user_id}")
                        log_authentication("Authenticated", user_id)
                        return user_id

    print("Unauthorized user")
    log_authentication("Unauthorized", user_id=None)
    send_twilio_notification()
    return None

def log_authentication(status, user_id):
    with open(LOG_FILE, 'a') as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Status: {status}, User ID: {user_id}\n")

if __name__ == "__main__":
    while True:
        print("Voice Authentication System")
        print("1. Collect Data")
        print("2. Train Model")
        print("3. Authenticate")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            collect_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            authenticate()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")