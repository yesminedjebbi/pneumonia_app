from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import models
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def build_model():
    base_model = ResNet50(
        include_top=False,
        weights=None,          # pas de weights ImageNet, on charge les nôtres
        input_shape=(256, 256, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



MODEL = build_model()
MODEL.load_weights("C:/Users/asus/Desktop/CNN_Project/pneumonia_app/backend/pneumonia_resnet50.h5")

print(" Modèle chargé avec succès !")


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))       
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)
    return image


@app.get("/")
def home():
    return {"message": "API Pneumonia Detection - OK"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)   # shape: (1, 256, 256, 3)

    predictions = MODEL.predict(img_batch)

    probability = float(predictions[0][0])
    predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]


    return {
        "prediction": predicted_class,
        "probability": round(probability, 4),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)