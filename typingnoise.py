#typing
import os
import random
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants for noise integration
NOISE_DATASET_PATH = "D:\neural\\noises"  # Path to your noise dataset
DATASET_PATH = "D:\\neural\\16000_pcm_speeches"
SAMPLE_RATE = 22050 
SNR_DB = 10  # Desired Signal-to-Noise Ratio in dB

# Function to load noise files
def load_noise_files():
    noise_files = []
    for root, _, files in os.walk(NOISE_DATASET_PATH):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                noise_files.append(file_path)
    return noise_files

# Function to add noise to an audio sample
def add_noise(audio):
    # Load a random noise file
    noise_file = random.choice(noise_files)
    noise, _ = librosa.load(noise_file, sr=SAMPLE_RATE)
    
    # Ensure the noise is the same length as the audio
    if len(noise) < len(audio):
        noise = librosa.util.fix_length(noise, size=len(audio))
    else:
        noise = noise[:len(audio)]

    # Calculate the RMS of the audio and noise
    audio_rms = np.sqrt(np.mean(audio**2))
    noise_rms = np.sqrt(np.mean(noise**2))

    # Calculate scaling factor for desired SNR
    target_noise_rms = audio_rms / (10 ** (SNR_DB / 20))
    scaling_factor = target_noise_rms / noise_rms if noise_rms > 0 else 0

    # Mix audio with scaled noise
    noisy_audio = audio + (noise * scaling_factor)
    
    return noisy_audio

# Load noise files
noise_files = load_noise_files()

# Update dataset preparation to include noisy samples
x_noisy, y_noisy = [], []
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            speaker_label = os.path.basename(root)

            # Extract clean features
            clean_features = extract_features(file_path)
            x_noisy.append(clean_features)
            y_noisy.append(speaker_label)

            # Add noise and extract features again
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            noisy_audio = add_noise(audio)
            noisy_features = extract_features(noisy_audio)
            x_noisy.append(noisy_features)
            y_noisy.append(speaker_label)

# Converting to numpy arrays
x_noisy = np.array(x_noisy)
y_noisy = np.array(y_noisy)

# Encode labels 
y_encoded_noisy = label_encoder.fit_transform(y_noisy) 
y_categorical_noisy = to_categorical(y_encoded_noisy) 

# Split dataset into training and testing for noisy data
x_train_noisy, x_test_noisy, y_train_noisy, y_test_noisy = train_test_split(x_noisy, y_categorical_noisy, test_size=0.2, random_state=42)
x_train_noisy = x_train_noisy[..., np.newaxis]
x_test_noisy = x_test_noisy[..., np.newaxis]

# Define and compile the model (if not already done)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on noisy data
history_noisy = model.fit(x_train_noisy, y_train_noisy, validation_data=(x_test_noisy, y_test_noisy), epochs=30, batch_size=32)

# Evaluate the model on noisy data
loss_noisy, accuracy_noisy = model.evaluate(x_test_noisy, y_test_noisy)
print(f"Test Accuracy on Noisy Data: {accuracy_noisy * 100:.2f}%")

# Make predictions on a test sample from noisy data
prediction_noisy = model.predict(x_test_noisy[:1])
predicted_index_noisy = np.argmax(prediction_noisy, axis=1)[0]
actual_index_noisy = np.argmax(y_test_noisy[0], axis=0)

predicted_speaker_noisy = label_encoder.classes_[predicted_index_noisy]
actual_speaker_noisy = label_encoder.classes_[actual_index_noisy]

print(f"\nPredicted Speaker (Noisy): {predicted_speaker_noisy}")
print(f"Actual Speaker (Noisy): {actual_speaker_noisy}")

# Save the updated model if needed
model.save("speaker_identification_model_with_noise.h5")
