import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()
# create mapping of image to captions

mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + \
                " ".join([word for word in caption.split()
                         if len(word) > 1]) + ' endseq'
            captions[i] = caption


clean(mapping)
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


@app.route('/')
def index():
    return render_template('index.html')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


model = tf.keras.models.load_model('model_50.h5')
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
        query = in_text
        stopwords = ['startseq', 'endseq']
        querywords = query.split()
        resultwords = [word for word in querywords if word.lower()
                       not in stopwords]
        result = ' '.join(resultwords)

    return result


print("="*50)
print('model loaded')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global model
    file = request.files['image']
    file.save('static/image.jpg')
    vgg_model = VGG16(weights='Vgg_model.h5')
    # restructure the model
    vgg_model = Model(inputs=vgg_model.inputs,
                      outputs=vgg_model.layers[-2].output)

    img = mpimg.imread('static/image.jpg')
    imgplot = plt.imshow(img)

    image_path = 'static/image.jpg'
    # load image
    image = load_img(image_path, target_size=(224, 224))

    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    feature = vgg_model.predict(image, verbose=0)
    # predict from the trained model

    final = predict_caption(model, feature, tokenizer, max_length)
    print(final)
    return render_template('predict.html', final=final)


if __name__ == "__main__":
    app.run(debug=True)
