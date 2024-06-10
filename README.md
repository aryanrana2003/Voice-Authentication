# Voice Authentication System

This project is a voice authentication system that uses machine learning to identify and authenticate users based on their voice. The system can collect voice samples, train a model, and authenticate users. Additionally, it sends a notification via Twilio if an unauthorized user attempts to access the system.

## Features

- **Voice Sample Collection**: Collects and stores voice samples from users.
- **Model Training**: Trains a Support Vector Machine (SVM) model for voice recognition.
- **Authentication**: Authenticates users based on their voice samples.
- **Anomaly Detection**: Uses Isolation Forest for anomaly detection to identify unauthorized access attempts.
- **Twilio Notification**: Sends an alert via SMS when an unauthorized access attempt is detected.

## Requirements

- Python 3.6+
- Libraries: `sounddevice`, `numpy`, `librosa`, `scikit-learn`, `twilio`
- Twilio account for SMS notifications

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/voice-authentication-system.git
    cd voice-authentication-system
    ```

2. Install the required libraries:
    ```sh
    pip install sounddevice numpy librosa scikit-learn twilio
    ```

3. Configure Twilio:
   - Sign up for a Twilio account and get your account SID, authentication token, and a Twilio phone number.
   - Update the Twilio configurations in the script:
     ```python
     TWILIO_ACCOUNT_SID = 'your_account_sid'
     TWILIO_AUTH_TOKEN = 'your_auth_token'
     TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
     RECIPIENT_PHONE_NUMBER = 'your_phone_number'
     ```

## Usage

1. Run the script:
    ```sh
    python voice_auth_system.py
    ```

2. Follow the on-screen instructions to:
   - **Collect Data**: Record voice samples for different users.
   - **Train Model**: Train the SVM model using the collected voice samples.
   - **Authenticate**: Authenticate a user by recording their voice and comparing it with the trained model.

3. If an unauthorized user attempts to access, a notification will be sent to the configured phone number.

## Code Overview

- `record_audio(duration, fs)`: Records audio for a specified duration and sampling rate.
- `extract_features(audio, sr)`: Extracts audio features (MFCC, chroma, mel spectrogram, spectral contrast).
- `augment_audio(audio, sr)`: Applies audio augmentation techniques (pitch shifting, time stretching, noise addition).
- `collect_data()`: Collects and saves augmented voice samples for a user.
- `train_model()`: Trains an SVM model and an Isolation Forest for voice authentication.
- `send_twilio_notification()`: Sends an SMS notification using Twilio.
- `authenticate()`: Authenticates a user based on their voice sample.
- `log_authentication(status, user_id)`: Logs authentication attempts.


## Acknowledgments

- [Librosa](https://librosa.org/) for audio processing.
- [Scikit-learn](https://scikit-learn.org/) for machine learning.
- [Twilio](https://www.twilio.com/) for SMS notifications.
