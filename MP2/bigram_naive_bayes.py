# bigram_naive_bayes.py
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


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.05, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=None, silently=False):
    pos_words = Counter()
    neg_words = Counter()
    pos_words_tot = 0
    neg_words_tot = 0

    pos_word_pairs = Counter()
    neg_word_pairs = Counter()
    pos_word_pairs_tot = 0
    neg_word_pairs_tot = 0  

    pos_reviews = 0
    
    for review,tag in tqdm(zip(train_set,train_labels), disable=silently):
        if tag:
            pos_reviews += 1

            pos_words_tot += len(review)
            pos_word_pairs_tot += len(review) - 1
            
            pos_words[review[0]] += 1
            for word_i in range(len(review)-1):
                pos_words[review[word_i + 1]] += 1
                pos_word_pairs[f"{review[word_i]},{review[word_i + 1]}"] += 1
        
        else:
            neg_words_tot += len(review)
            neg_word_pairs_tot += len(review) - 1

            neg_words[review[0]] += 1
            for word_i in range(len(review)-1):
                neg_words[review[word_i + 1]] += 1
                neg_word_pairs[f"{review[word_i]},{review[word_i + 1]}"] += 1


    if pos_prior == None:
        pos_prior = pos_reviews/len(train_set)
    if not silently:    
        print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    yhats = []
    pos_prior_log = math.log(pos_prior)
    neg_prior_log = math.log(1 - pos_prior)
    pos_denom_uni = pos_words_tot + unigram_laplace * (len(pos_words) + 1)
    neg_denom_uni = neg_words_tot + unigram_laplace * (len(neg_words) + 1)
    pos_denom_bi = pos_word_pairs_tot + bigram_laplace * (len(pos_word_pairs) + 1)
    neg_denom_bi = neg_word_pairs_tot + bigram_laplace * (len(neg_word_pairs) + 1)

    for doc in tqdm(dev_set, disable=silently):
        pos_prob_uni = pos_prob_bi = pos_prior_log
        neg_prob_uni = neg_prob_bi = neg_prior_log

        pos_prob_uni += math.log((pos_words[doc[0]] + unigram_laplace)/pos_denom_uni)
        neg_prob_uni += math.log((neg_words[doc[0]] + unigram_laplace)/neg_denom_uni)
        for word_i in range(len(doc)-1):
            pos_prob_uni += math.log((pos_words[doc[word_i + 1]] + unigram_laplace)/pos_denom_uni)
            neg_prob_uni += math.log((neg_words[doc[word_i + 1]] + unigram_laplace)/neg_denom_uni)

            pos_prob_bi += math.log((pos_word_pairs[f"{doc[word_i]},{doc[word_i + 1]}"] + bigram_laplace)/pos_denom_bi)
            neg_prob_bi += math.log((neg_word_pairs[f"{doc[word_i]},{doc[word_i + 1]}"] + bigram_laplace)/neg_denom_bi)

        pos_prob = bigram_lambda * pos_prob_bi + (1 - bigram_lambda) * pos_prob_uni
        neg_prob = bigram_lambda * neg_prob_bi + (1 - bigram_lambda) * neg_prob_uni

        yhats.append(1 if pos_prob > neg_prob else 0)

    return yhats



