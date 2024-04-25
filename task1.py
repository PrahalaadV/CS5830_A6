from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import keras.models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import uvicorn
from keras.layers import InputLayer


app = FastAPI()

## Parse path to the model from command line
def parse_command():

    parser = argparse.ArgumentParser(description='Load model')
    parser.add_argument('path',type=str, help="Path of the model")
    
    ## parse the arguements
    args = parser.parse_args()

    return args.path

## Load the model
def load_model(path):
  model = keras.models.load_model(path)
  return model

## Predict digit given the input image
def predict_digit(model, img):
    prediction = model.predict(img.reshape(1,-1))
    return np.argmax(prediction)

## API endpoint /predict - Reads input image and converts it into a serialized array though an asynchronous function
@app.post('/predict')
async def predict(upload_file: UploadFile = File(...)):
    
    # Read the uploaded image file
    contents = await upload_file.read()
    
    # Convert the uploaded image to grayscale and resize
    img = Image.open(io.BytesIO(contents))
    
    # Convert image to numpy array
    img_array = np.array(img.convert('L'))/255 
    
    # Flatten the image array
    #img_flattened = img_array.flatten()
    
    # Obtain path from command line and load the model
    path = parse_command()
    model = load_model(path)
    
    # Predict the digit
    digit = predict_digit(model, img_array)
    
    # Return the predicted digit as JSON response
    return {"digit": str(digit)}

if __name__ == '__main__':
    uvicorn.run("task1:app", reload=True)