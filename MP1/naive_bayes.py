# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
import nltk
import numpy as np


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.6, silently=False):
    print_values(laplace,pos_prior)
    
    pos_words = Counter()
    neg_words = Counter()
    pos_words_tot = 0
    neg_words_tot = 0
    
    for review,tag in tqdm(zip(train_set,train_labels), disable=silently):
        if tag:
            pos_words_tot += len(review)
            pos_words = pos_words + Counter(review)
        else:
            neg_words_tot += len(review)
            neg_words = neg_words + Counter(review)

    print(pos_words_tot)
    print(neg_words_tot)

    yhats = []
    pos_prior_log = math.log(pos_prior)
    neg_prior_log = math.log(1 - pos_prior)
    pos_denom = pos_words_tot + laplace * len(pos_words)+1
    neg_denom = neg_words_tot + laplace * len(neg_words)+1

    for doc in tqdm(dev_set, disable=silently):
        pos_prob = pos_prior_log
        neg_prob = neg_prior_log
        for word in doc:
            pos_prob += math.log((pos_words[word] + laplace)/pos_denom)
            neg_prob += math.log((neg_words[word] + laplace)/neg_denom)

        yhats.append(1 if pos_prob > neg_prob else 0)

    return yhats



