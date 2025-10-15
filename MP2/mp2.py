# mp2.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023

import sys
import argparse
import configparser
import copy

import reader
import bigram_naive_bayes as nb

"""
This file contains the main application that is run for this MP.
"""

def compute_accuracies(predicted_labels, dev_labels):
    yhats = predicted_labels
    assert len(yhats) == len(dev_labels), "predicted and gold label lists have different lengths"
    accuracy = sum([yhats[i] == dev_labels[i] for i in range(len(yhats))]) / len(yhats)
    tp = sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    tn = sum([yhats[i] == dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))])
    fp = sum([yhats[i] != dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    fn = sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))])
    return accuracy, fp, fn, tp, tn

# print value and also percentage out of n
def print_value(label, value, numvalues):
   print(f"{label} {value} ({value/numvalues * 100}%)")

# print out performance stats
def print_stats(accuracy, false_positive, false_negative, true_positive, true_negative, numvalues):
    print(f"Accuracy: {accuracy}")
    print_value("False Positive", false_positive,numvalues)
    print_value("False Negative", false_negative,numvalues)
    print_value("True Positive", true_positive,numvalues)
    print_value("True Negative", true_negative,numvalues)
    print(f"total number of samples {numvalues}")


"""
Main function
    You can modify the default parameter settings given below, 
    instead of constantly typing your favorite values at the command line.
"""
def main(args):
    train_set, train_labels, dev_set, dev_labels = nb.load_data(args.training_dir,args.development_dir,args.stemming,args.lowercase)
    
    lap_uni = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    lap_bi = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    bigram_lambda = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    max_accuracy = 0
    max_stats = ()
    index = 0

    for u in lap_uni:
        for b in lap_bi:
            for bl in bigram_lambda:
                index += 1
                predicted_labels = nb.bigram_bayes(train_set, train_labels, dev_set,
                                                    u,b,bl,0.5, silently=True)

                accuracy, false_positive, false_negative, true_positive, true_negative = compute_accuracies(predicted_labels,dev_labels)
                nn = len(dev_labels)
                print(f"{index}: {accuracy}")

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_stats = (u,b,bl)

    print("Best results:")
    print("----------------")
    print(f"Unigram Laplace: {max_stats[0]}")
    print(f"Bigram Laplace: {max_stats[1]}")
    print(f"Bigram Lambda: {max_stats[2]}")
    print(f"Positive prior: {0.5}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP2 Bigram Naive Bayes')
    parser.add_argument('--training', dest='training_dir', type=str, default = 'data/movie_reviews/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = 'data/movie_reviews/dev',
                        help='the directory of the development data')

    # When doing final testing, reset the default values below to match your settings in naive_bayes.py
    parser.add_argument('--stemming',dest="stemming", type=bool, default=False,
                        help='Use porter stemmer')
    parser.add_argument('--lowercase',dest="lowercase", type=bool, default=False,
                        help='Convert all word to lower case')
    parser.add_argument('--laplace',dest="laplace", type=float, default = 1.0,
                        help='Laplace smoothing parameter')
    parser.add_argument('--bigram_laplace',dest="bigram_laplace", type=float, default = 1.0,
                        help='Laplace smoothing parameter for bigrams')
    parser.add_argument('--bigram_lambda',dest="bigram_lambda", type=float, default = 1.0,
                        help='Weight on bigrams vs. unigrams')
    parser.add_argument('--pos_prior',dest="pos_prior", type=float, default = 0.5,
                        help='Positive prior, i.e. percentage of test examples that are positive')

    args = parser.parse_args()
    main(args)
