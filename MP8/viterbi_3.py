"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict
from math import log,exp
from copy import deepcopy


# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect

numerals = {"0","1","2","3","4","5","6","7","8","9"}
word_classes = ["NUMERICAL", "SHORT", "LONG-S", "LONG", "MED-S", "MED"]
def classfiy_word(word):
    if word[0] in numerals and word[-1] in numerals:
        return "NUMERICAL"
    if len(word) <= 3:
        return "SHORT"
    
    if len(word) >= 10:
        if word[-1].lower() == "s":
            return "LONG-S"
        return "LONG"
              
    if word[-1].lower() == "s":
        return "MED-S"
    return "MED"


def training_v3(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    hapax_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word_class: # }}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    emit_prob_r = defaultdict(lambda: defaultdict(lambda: 0)) # {word:{tag: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    for sentence in sentences:
        init_prob[sentence[0][1]] += 1
        emit_prob[sentence[0][1]][sentence[0][0]] += 1
        emit_prob_r[sentence[0][0]][sentence[0][1]] += 1

        for word_idx in range(1, len(sentence)):
            word = sentence[word_idx][0]
            tag_0 = sentence[word_idx - 1][1]
            tag_1 = sentence[word_idx][1]

            emit_prob[tag_1][word] += 1
            emit_prob_r[word][tag_1] += 1
            trans_prob[tag_0][tag_1] += 1

    for word, tags in emit_prob_r.items():
        if sum(tags.values()) == 1:
            hapax_prob[list(tags.keys())[0]][classfiy_word(word)] += 1

    tag_list = list(emit_prob.keys())
    init_prob_nT = sum(init_prob.values())
    hapax_prob_nT = (sum([sum(tag.values()) for tag in hapax_prob.values()]) + 
                     (len(tag_list) * len(word_classes) - sum([len(tag.keys()) for tag in hapax_prob.values()])))
    for t1 in tag_list:

        for word_class in word_classes:
            if t1 in hapax_prob and word_class in hapax_prob[t1]:
                hapax_prob[t1][word_class] = (hapax_prob[t1][word_class]/hapax_prob_nT)
            else:
                hapax_prob[t1][word_class] = 1.0/hapax_prob_nT
        
        if t1 in init_prob:
            init_prob[t1] = (init_prob[t1]/init_prob_nT)
        else:
            init_prob[t1] = epsilon_for_pt

        if t1 not in trans_prob:
           for t2 in tag_list:
               trans_prob[t1][t2] = 1/len(tag_list)
        else:
            trans_prob_nT = sum(trans_prob[t1].values())
            trans_prob_VT = len(trans_prob[t1])
            for t2 in tag_list:
                if t2 in trans_prob[t1]:
                    trans_prob[t1][t2] = (trans_prob[t1][t2] + epsilon_for_pt) / (trans_prob_nT + epsilon_for_pt * (trans_prob_VT + 1))
                else:
                    trans_prob[t1][t2] =((epsilon_for_pt) / (trans_prob_nT + epsilon_for_pt * (trans_prob_VT + 1))) * ((len(tag_list) - trans_prob_VT)/len(tag_list))

        emit_prob_nT = sum(emit_prob[t1].values())
        emit_prob_VT = len(emit_prob[t1])
        for word in emit_prob[t1]:
            emit_prob[t1][word] = (emit_prob[t1][word] + emit_epsilon * sum(hapax_prob[t1].values())) / (emit_prob_nT + emit_epsilon * sum(hapax_prob[t1].values()) * (emit_prob_VT + 1))
        for word in word_classes:
            emit_prob[t1][word] = (emit_epsilon * hapax_prob[t1][word]) / (emit_prob_nT + emit_epsilon * hapax_prob[t1][word] * (emit_prob_VT + 1))
            
    
    return init_prob, emit_prob, trans_prob

def viterbi_3_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    for t1 in prev_prob:
        if word in emit_prob[t1]:
            log_emit_prob = log(emit_prob[t1][word])
        else:
            log_emit_prob = log(emit_prob[t1][classfiy_word(word)])

        max_prob = -math.inf
        max_tag = ""
        for t0, log_prev_prob in prev_prob.items():
            running_prob = log_prev_prob + log_emit_prob

            if i != 0:
                running_prob += log(trans_prob[t0][t1])

            if running_prob > max_prob:
                max_prob = running_prob
                max_tag = t0

        log_prob[t1] = max_prob
        if i != 0:
            predict_tag_seq[t1] = deepcopy(prev_predict_tag_seq[max_tag])
            predict_tag_seq[t1].append(max_tag)
        else:
            predict_tag_seq = prev_predict_tag_seq

    return log_prob, predict_tag_seq

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    init_prob, emit_prob, trans_prob = training_v3(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_3_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        max_prob = -math.inf
        max_seq = []
        for t, prob in log_prob.items():
            if prob > max_prob:
                max_prob = prob
                max_seq = predict_tag_seq[t]
                max_seq.append(t)

        predicts.append(list(zip(sentence, max_seq)))      

    return predicts