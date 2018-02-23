# coding: utf-8

import os
import nltk 
import random

from collections import Counter 
from nltk import word_tokenize, WordNetLemmatizer, \
                 NaiveBayesClassifier, classify
from nltk.corpus import stopwords

def init_lists(folder):
    """ Read the files from the dataset folder """
    files_list = []
    folders = os.listdir(folder)
    for file in folders:
        file = open(folder + file, 'r', errors='ignore')
        files_list.append(file.read())
    file.close()
    return files_list

def preprocess(sentece):
    """ 
    Preprocess the data splitting the words and linking
    the different form of the same word
    """
    tokens = word_tokenize(sentece)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in tokens]

def get_features(text, setting):
    if setting == 'bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stopwords_list}
    else:
        return {word: True for word  in preprocess(text) if not word in stopwords_list}

def train(features, samples_proportion):
    """ 
    Initialise the training and test sets 
    and train the model
    """
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    
    print('Training set size = {0}'.format(len(train_set)))
    print('Test set size = {0}'.format(len(test_set)))
    
    classifier = NaiveBayesClassifier.train(train_set)
    
    return train_set, test_set, classifier 

def evaluate(trainset, test_set, classifier):
    """ Calculates the accuracy of our classifier """
    print('Accuracy of the training set = {0}'.format(
        classify.accuracy(classifier, train_set)
    ))
    print('Accuracy of the test set = {0}'.format(
        classify.accuracy(classifier, test_set)
    ))

    #classifier.show_most_informative_features(20)

if __name__ == '__main__':
    spam = init_lists('datasets/enron1/spam/')
    ham = init_lists('datasets/enron1/ham/')

    spam_emails = [(email, 'spam') for email in spam]
    ham_emails = [(email, 'ham') for email in ham]
    all_emails = spam_emails + ham_emails
    
    random.shuffle(all_emails)
    
    stopwords_list = stopwords.words('english')
    
    all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]

    train_set, test_set, classifier = train(all_features, 0.8)

    print(evaluate(train_set, test_set, classifier))
    