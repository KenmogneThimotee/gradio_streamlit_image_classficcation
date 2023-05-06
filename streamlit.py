
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import numpy as np


def load_model() :
    model = tf.keras.models.load_model('model.h5')

    return model



if __name__ == '__main__':
    
    # Get the model
    model = load_model()

    # Load labels
    birds_csv = pd.read_csv("archive/birds.csv")
    labels = list(set(birds_csv.labels))
    

    st.title('BIRDS 525 SPECIES IMAGE CLASSIFICATION')


    file = st.file_uploader('Upload An Image', type=['png', 'jpg'])

    if file: 
        
        st.write("Your Image")
        st.image(file)
        
        # Decode image to numpy array
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
        img = np.array(img)

        
         # Format the image 
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        
        # Predict bird speci
        prediction = model(img)
        
        # Assign prediction with labels
        prediction = prediction[0]
        confidences = {labels[i]: float(prediction[i] * 100) for i in range(525)}
        
        # Sort the dictionary by values.
        sorted_dictionary = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

        # Get the first 10 elements of the sorted dictionary.
        first_10_elements = sorted_dictionary[:10]
        
        # Display the most 10 labels
        st.bar_chart(dict(map(lambda x: (x[0], x[1]), first_10_elements)),
                     width=100,
                     height=500)