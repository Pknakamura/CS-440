"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict
from collections import Counter


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_dict = defaultdict(lambda: Counter())
    all_tags = Counter()
    for sentence in train:
        for word, tag in sentence:
            word_dict[word][tag] += 1
            all_tags[tag] += 1

    default_tag = all_tags.most_common(1)[0][0]
    tagged_list = []

    for sentence in test:
        tagged_sentence = []
        for word in sentence:

            if word in word_dict:
                tag = word_dict[word].most_common(1)[0][0]

            else:
                tag = default_tag

            tagged_sentence.append((word, tag))

        tagged_list.append(tagged_sentence)

    return tagged_list
