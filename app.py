# Basic imports
import text_hammer as th
import numpy as np

# Importing transformer
from transformers import AutoTokenizer,TFBertModel

# Importing tensorflow and layers
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

# Flask utils
from flask import Flask, request, render_template

# For production environment
from gevent import pywsgi

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'best_model.h5'
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

# Load your trained model
def model_load(path) :
    max_len = 170

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = Dense(128, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = Dense(32,activation = 'relu')(out)

    y = Dense(2,activation = 'sigmoid')(out)
        
    new_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    new_model.layers[2].trainable = True
    # for training bert our lr must be so small

    new_model.load_weights(path)
    return new_model

new_model = model_load(MODEL_PATH)

# Printing that the model has loaded
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')

# Text preprocessing function
def text_preprocessing(text):
    text = str(text).lower()
    text = th.cont_exp(text) #you're -> you are; i'm -> i am
    text = th.remove_emails(text)
    text = th.remove_html_tags(text)
    #     text = ps.remove_stopwords(text)
    text = th.remove_special_chars(text)
    text = th.remove_accented_chars(text)
    #     text = th.make_base(text) #ran -> run,
    return text

# Predicting sentiment
def predict_sentiment(texts, new_model):
    texts = text_preprocessing(texts)
    x_val = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=170,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True) 
    validation = new_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    classes = ['The Text Does NOT Contains References to Self-Harm ✅', 'The Text Contains References to Self-Harm🚩']
    sentiment_predicted = classes[np.argmax(validation[0])]
    return sentiment_predicted

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get query from post request
        my_query = next(request.form.items())[0]
        
        # Make prediction
        result = predict_sentiment(my_query, new_model)
        return result
    return None


if __name__ == '__main__':
    # Uncomment when going for production
    # server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    # server.serve_forever()
    app.run(debug=True)