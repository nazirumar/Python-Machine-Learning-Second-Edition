

import os
import re
import pickle
from sklearn.feature_extraction.text import HashingVectorizer

# Get the current directory
cur_dir = os.path.dirname(__file__)

# Load stopwords from the pickle file
stopwords_path = os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl')
try:
    with open(stopwords_path, 'rb') as stopwords_file:
        stop = pickle.load(stopwords_file)
except FileNotFoundError:
    print(f"Stopwords file not found at {stopwords_path}")
    stop = []

# Define the tokenizer function
def tokenizer(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    # Remove non-word characters and append emoticons
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    # Tokenize text and remove stopwords
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Initialize the HashingVectorizer
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)



# from sklearn.feature_extraction.text import HashingVectorizer
# import re
# import os
# import pickle


# cur_dir = os.path.dirname(__file__)

# stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))


# def tokenizer(text):
#     text = re.sub('<[^>]*>', '', text)
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
#     text.lower())
#     text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
#     tokenized = [w for w in text.split() if w not in stop]
#     return tokenized


# vect = HashingVectorizer(decode_error='ignore',
#                             n_features=2**21,
#                             preprocessor=None,
#                             tokenizer=tokenizer)

