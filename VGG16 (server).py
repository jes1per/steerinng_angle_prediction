import pickle
import os
import time
import gc
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# Define constant paths
DATA_PATH = '/home/m_mirzaabbasi/Sully_Chen_Training/preprocessed_sully_chen_vgg16.h5'               # CHECK THIS
MODEL_PATH = '/home/m_mirzaabbasi/Sully_Chen_Training/models/vgg16_server'                      # CHECK THIS
JSON_PATH = '/home/m_mirzaabbasi/Sully_Chen_Training/logs/vgg16_server'      # CHECK THIS
VALIDATION_PATH = '/home/m_mirzaabbasi/Sully_Chen_Training/validation/vgg16_server'      # CHECK THIS

def save_best_val_loss(val_loss, split_num, epoch_num):
    with open(os.path.join(JSON_PATH, f'training_history_epoch_{epoch_num}.json'), 'w') as f:
        json.dump({'best_val_loss': val_loss, 'last_split': split_num}, f)

def load_best_val_loss(epoch_num):
    try:
        with open(os.path.join(JSON_PATH, f'training_history_epoch_{epoch_num}.json'), 'r') as f:
            data = json.load(f)
            return data['best_val_loss']
    except:
        return float('inf')

def load_split(split_number):
    print(f"Loading split {split_number}...")
    file_path = os.path.join(DATA_PATH, f'preprocessed_split_{split_number}.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Split {split_number} loaded successfully")
    return data[0], data[1]
def create_vgg13():
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))   # Changed on this training
    model.add(Dropout(0.1))                     # Changed on this training
    model.add(Dense(1024, activation='relu'))   # Changed on this training
    model.add(Dropout(0.1))                     # Changed on this training
    model.add(Dense(1))

    return model
class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, initial_best_val_loss):
        super().__init__()
        self.filepath = filepath
        self.best_val_loss = initial_best_val_loss

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            print(f'\nEpoch {epoch + 1}: val_loss improved from {self.best_val_loss} to {current_val_loss}, saving model to {self.filepath}')
            self.best_val_loss = current_val_loss
            self.model.save(self.filepath)

def train_model_on_split(model, split_num, epoch_num, epochs=10, previous_best_val_loss=float('inf')):
    print(f"\n{'='*50}")
    print(f"Processing split {split_num}")
    print(f"{'='*50}\n")
    print(f"Previous best validation loss: {previous_best_val_loss}")

    X_train, y_train = load_split(split_num)

    checkpoint_path = os.path.join(MODEL_PATH, f'model_checkpoint_epoch_{epoch_num}_split_{split_num}.keras')
    custom_checkpoint = CustomModelCheckpoint(checkpoint_path, previous_best_val_loss)

    start_time = time.time()

    print(f"Starting training epoch {epoch_num} on split {split_num}")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        batch_size=32,
        callbacks=[custom_checkpoint],
        verbose=1
    )

    best_val_loss = min(history.history['val_loss'])

    training_time = time.time() - start_time
    print(f"\nTraining time for epoch {epoch_num} split {split_num}: {training_time:.2f} seconds")

    final_model_path = os.path.join(MODEL_PATH, f'model_final_epoch_{epoch_num}_split_{split_num}.keras')
    model.save(final_model_path)

    if best_val_loss < previous_best_val_loss:
        print(f"Model saved after epoch {epoch_num} split {split_num} (Improved val_loss: {best_val_loss})")
        save_best_val_loss(best_val_loss, split_num, epoch_num)
    else:
        print(f"Model saved after epoch {epoch_num} split {split_num} (Maintaining previous best val_loss: {previous_best_val_loss})")
        save_best_val_loss(previous_best_val_loss, split_num, epoch_num)

    del X_train
    del y_train
    import gc
    gc.collect()

    return min(best_val_loss, previous_best_val_loss)
def model_train(SPLIT_TO_TRAIN, EPOCH_TO_TRAIN):
    EPOCHS = 1
    # Create model directory
    os.makedirs(MODEL_PATH, exist_ok=True)

    if SPLIT_TO_TRAIN == 1 and EPOCH_TO_TRAIN == 1:
        print("Starting fresh training for split 1")
        model = create_vgg13()
        best_val_loss = float('inf')

    elif SPLIT_TO_TRAIN == 1 and EPOCH_TO_TRAIN > 1:
        previous_split = 8
        previous_epoch = EPOCH_TO_TRAIN - 1
        previous_model_path = os.path.join(MODEL_PATH, f'model_final_epoch_{previous_epoch}_split_{previous_split}.keras')

        if not os.path.exists(previous_model_path):
            raise ValueError(f"Cannot find model for epoch {previous_epoch} split {previous_split}. Please train on previous split first.")

        print(f"Loading model from epoch {previous_epoch} split {previous_split}")
        model = load_model(previous_model_path)
        best_val_loss = load_best_val_loss(previous_epoch)

    else:
        previous_split = SPLIT_TO_TRAIN - 1
        previous_model_path = os.path.join(MODEL_PATH, f'model_final_epoch_{EPOCH_TO_TRAIN}_split_{previous_split}.keras')

        if not os.path.exists(previous_model_path):
            raise ValueError(f"Cannot find model for epoch {EPOCH_TO_TRAIN} split {previous_split}. Please train on previous split first.")

        print(f"Loading model from epoch {EPOCH_TO_TRAIN} split {previous_split}")
        model = load_model(previous_model_path)
        best_val_loss = load_best_val_loss(EPOCH_TO_TRAIN)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Train on the target split
    best_val_loss = train_model_on_split(model, SPLIT_TO_TRAIN, EPOCH_TO_TRAIN, EPOCHS, best_val_loss)
    print(f"\nTraining completed for epoch {EPOCH_TO_TRAIN} split {SPLIT_TO_TRAIN}!")

def validate_model(EPOCH_TO_TRAIN):
    # Load the final trained model from split 8
    print("Loading best model from training splits...")
    final_model = load_model(os.path.join(MODEL_PATH, f'model_final_epoch_{EPOCH_TO_TRAIN}_split_8.keras'))

    # Load validation data (split 9)
    print(f"\nLoading validation epoch {EPOCH_TO_TRAIN} split (split 9)...")
    X_val, y_val = load_split(9)

    # Evaluate model on validation split
    print("\nEvaluating model on validation split...")
    validation_results = final_model.evaluate(X_val, y_val, verbose=1)
    print(f"\nValidation Results:")
    print(f"Loss: {validation_results[0]:.4f}")
    print(f"MAE: {validation_results[1]:.4f}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = final_model.predict(X_val, verbose=1)

    # Calculate and display additional metrics
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, predictions)

    print("\nAdditional Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Clean up memory
    del X_val, y_val
    gc.collect()

    # Create a dictionary with all validation metrics
    validation_metrics = {
        'loss': float(validation_results[0]),
        'mae': float(validation_results[1]),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }

    # Save metrics to a JSON file
    validation_results_path = os.path.join(VALIDATION_PATH, f'validation_results_epoch_{EPOCH_TO_TRAIN}_split_9.json')
    with open(validation_results_path, 'w') as f:
        json.dump(validation_metrics, f, indent=4)

    # Optionally, save predictions to a numpy file
    predictions_path = os.path.join(VALIDATION_PATH, f'validation_predictions_epoch_{EPOCH_TO_TRAIN}_split_9.npy')
    np.save(predictions_path, predictions)

    print(f"Validation results saved to: {validation_results_path}")
    print(f"Predictions saved to: {predictions_path}")
