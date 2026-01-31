import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps

MODEL_PATH = Path("models/mnist.keras")

def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def ensure_model() -> tf.keras.Model:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)

    # Train quick baseline
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    model = build_model()
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Saved model. Test accuracy: {test_acc:.4f}")

    model.save(MODEL_PATH)
    return model

MODEL = ensure_model()

def preprocess_pil_to_mnist(pil_img: Image.Image) -> np.ndarray:
    """
    Gradio tekent typisch donker op licht. MNIST verwacht licht op donker.
    Dus: grayscale -> invert -> resize -> normalize -> shape (1, 28, 28)
    """
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = np.array(img).astype("float32") / 255.0
    x = x.reshape(1, 28, 28)
    return x

def predict_from_editor(payload):
    """
    gr.ImageEditor geeft meestal een dict terug met o.a. 'composite' (PIL).
    We ondersteunen ook direct PIL input (voor compatibiliteit).
    """
    if isinstance(payload, dict) and "composite" in payload and payload["composite"] is not None:
        pil_img = payload["composite"]
    elif isinstance(payload, Image.Image):
        pil_img = payload
    else:
        return {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0}

    x = preprocess_pil_to_mnist(pil_img)
    probs = MODEL.predict(x, verbose=0)[0]
    return {str(i): float(probs[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict_from_editor,
    inputs=gr.ImageEditor(type="pil", crop_size="1:1", label="Teken een cijfer (0-9)"),
    outputs=gr.Label(num_top_classes=3, label="Voorspelling"),
    title="MNIST – Teken & Voorspel (TensorFlow)",
    description="Teken een cijfer. De app zet het om naar 28×28 zoals MNIST en voorspelt met het model."
)

if __name__ == "__main__":
    # In Codespaces: bind op 0.0.0.0 zodat port forwarding werkt
    demo.launch(server_name="0.0.0.0", server_port=7860)
