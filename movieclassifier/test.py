# import pickle
# import re
# import os
# from vectorizer import vect
# import numpy as np

# clf = pickle.load(open(os.path.join('pkl_objects','classifier.pkl'), 'rb'))


# label = {0:'negative', 1:'positive'}

# example = ['I love this movie']

# X = vect.transform(example)

# print('Prediction: %s\n Probability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))

import re
import pickle
import os
import numpy as np
from sklearn.exceptions import NotFittedError

# Ensure this import is present
from sklearn.feature_extraction.text import HashingVectorizer

# Load stopwords (assuming you have the tokenizer function defined as in the previous script)
cur_dir = os.path.dirname(__file__)
stopwords_path = os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl')

try:
    with open(stopwords_path, 'rb') as stopwords_file:
        stop = pickle.load(stopwords_file)
except FileNotFoundError:
    print(f"Stopwords file not found at {stopwords_path}")
    stop = []

# Define the tokenizer function
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Initialize the HashingVectorizer
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

try:
    # Load the classifier
    with open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb') as model_file:
        clf = pickle.load(model_file)

    # Define label mapping
    label = {0: 'negative', 1: 'positive'}

    # Example to predict
    example = ['I love this movie']

    # Transform the example using the vectorizer
    X = vect.transform(example)

    # Make prediction and get probability
    prediction = clf.predict(X)[0]
    probability = np.max(clf.predict_proba(X)) * 100

    # Print the results
    print('Prediction: %s\nProbability: %.2f%%' % (label[prediction], probability))

except NotFittedError as e:
    print("Model not fitted: ", e)
except FileNotFoundError:
    print("Classifier file not found.")
except Exception as e:
    print("An error occurred: ", e)
