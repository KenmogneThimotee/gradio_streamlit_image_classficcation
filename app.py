import tensorflow as tf
import gradio as gr
import requests
import pandas as  pd
import numpy as np

model = tf.keras.models.load_model('model.h5')
birds_csv = pd.read_csv("archive/birds.csv")

labels = list(set(birds_csv.labels))


def classify_image(img):
  
  
  # Format the image 
  img = np.array(img)
  img = np.expand_dims(img, axis=0)

  # Predict bird speci
  prediction = model(img)
  
  # Assign prediction with labels
  prediction = prediction.numpy().tolist()
  prediction = prediction[0]
  confidences = {labels[i]: float(prediction[i]) for i in range(525)}
  
  return confidences


# Create a gradio interface
gr.Interface(fn=classify_image, 
             inputs=gr.Image(shape=(224, 224)),
             outputs=gr.Label(num_top_classes=10),
             title="BIRDS 525 SPECIES IMAGE CLASSIFICATION").launch()
