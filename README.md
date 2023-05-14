# Next Word Prediction

Live Demo :- https://webappnextword.azurewebsites.net

This is a Flask application that uses a trained LSTM model to predict the next words given a prompt.

## Project Overview

This project is built using TensorFlow and Flask. It uses a trained LSTM model to predict the next words given a prompt. The training data for the model is from a corpus of text. The model is trained to predict the probability distribution of the next word given the previous words. 

## Model

The model is an LSTM network with an embedding layer. The model is trained to predict the probability distribution of the next word given the previous words. The input sequence is fed into the embedding layer and then into the LSTM layer. The output of the LSTM layer is then passed through a dense layer with softmax activation function to get the probability distribution of the next word.

## Flask App

The Flask app consists of two routes. The first route, `/`, is the home page which contains a form for the user to enter a prompt. The second route, `/predict`, takes the user's input, processes it using the trained model, and returns the predicted next words.

## Screenshots
<img src="https://github.com/Sanket1909/nextwordpro/blob/main/main.png" alt="screenshot 1" width="600" height="300"> <img src="https://github.com/Sanket1909/nextwordpro/blob/main/result.png" alt="screenshot 2" width="600" height="300">


## Getting Started

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- numpy

### Installation

1. Clone the repository
2. Install the required libraries using `pip install -r requirements.txt`
3. Run the Flask app using `python app.py`

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
