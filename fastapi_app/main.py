from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = FastAPI()

model = load_model('App_model.h5')
image_size = (224, 224)

def preprocess_image(img):
    img = img.resize(image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def predict_image(img):
    x = preprocess_image(img)
    predictions = model.predict(x)
    class_names = ["cleaning", "gardening", "handyman", "pet"]
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence_rate = predictions[0][predicted_class_index] * 100
    return predicted_class_name, confidence_rate

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    predicted_class, confidence_rate = predict_image(img)
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence_rate": confidence_rate
    })
