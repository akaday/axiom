import tensorflow as tf

def run_ai_task():
    """
    Function to run AI tasks using TensorFlow.
    """
    print("Running AI task with TensorFlow")
    # Example code for TensorFlow model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    print("AI task completed.")

if __name__ == "__main__":
    run_ai_task()
