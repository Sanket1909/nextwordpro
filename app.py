from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical

app = Flask(__name__,template_folder='templates')

# Load the saved model and other data
new_model = load_model('model.h5')
tokens = text.Tokenizer()
max_length = 11  # Change this to the maximum length of input sequence
with open('dataset.txt', 'r') as fp:
    data = fp.read().splitlines()
tokens.fit_on_texts(data)
vocab_size = len(tokens.word_counts) + 1
idx2word = {v:k for k,v in tokens.word_index.items()}

# Function to predict the next words
def predict_words(text, num_words=3):
    encoded_data = tokens.texts_to_sequences([text])[0]
    padded_data = sequence.pad_sequences([encoded_data], maxlen=max_length+1, padding='pre')
    y_preds = new_model.predict(padded_data)
    y_preds = np.argsort(-y_preds)
    y_preds = y_preds[0][:num_words]
    possible_words = [idx2word[item] for item in y_preds]
    return possible_words

# Routes
@app.route('/')
def home():
    return render_template('/home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    predictions = predict_words(text)
    return render_template('result.html', text=text, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
