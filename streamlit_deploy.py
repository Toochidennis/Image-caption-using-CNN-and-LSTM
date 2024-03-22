from tabnanny import verbose
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras.applications import DenseNet201
from textwrap import wrap


# model path
working_dir = 'working/'

with open(os.path.join(working_dir, 'descriptions.txt'), 'r') as file:
    caption_lines = file.readlines()

captions = [line.strip() for line in caption_lines]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

base_model = load_model('working/model.h5')

max_length = max(len(caption.split()) for caption in captions)


def idx_to_word(integer,tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

def generate_caption(image):
    model = DenseNet201()
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    # load image
    image = load_img(image, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image/255.
    image =np.expand_dims(image, axis=0)
    feature = fe.predict(image, verbose=0)
    # predict from the trained model
    caption = predict_caption(base_model, feature, tokenizer, 34)
    return caption


# Streamlit app
def main():
    st.title('Image Captioning with Streamlit')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Generating caption...")
        caption = generate_caption(uploaded_file)
        st.write("Caption:", caption)

if __name__ == '__main__':
    main()