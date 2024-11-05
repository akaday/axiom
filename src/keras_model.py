import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Define the Keras model using tf.keras.Input for the input shape
model = Sequential([
    tf.keras.Input(shape=(28, 28, 1)),  # Correct way to specify input shape
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Example function to run the AI task
def run_ai_task():
    """
    Function to run AI tasks using TensorFlow.
    """
    print("Running AI task with TensorFlow")
    # Example code to use the TensorFlow model
    print("AI task completed.")

if __name__ == "__main__":
    run_ai_task()
