import nltk                               # for creating a lexion
from nltk.tokenize import word_tokenize   # tokenize each word
from nltk.stem import WordNetLemmatizer   # word lemmatizer for treating the variations of a word as a sinlge word
import numpy as np
import pickle
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

# Function to create a lexicon (dictionary) of all the words in pos and neg text files
def create_lexicon(pos,neg):
    lexicon = []
    for sent in [pos,neg]:
        with open(sent,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    
    #Excluding the very often and very rare used words
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print('Size of input feature', len(l2))
    return l2

# Function to create the featureset with the lexicon
def sample_handling(sample, lexicon, classification):
    featureset = []
    
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features,classification])
    return featureset

# Function to create the feature sets and labels
def create_feature_sets_and_labels(pos,neg,test_size =0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    
    features = np.array(features)
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
