
# -*- coding: ascii -*-
import os, sys

import pandas as pd

import csv

import string

import nltk

from time import time
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from functools32 import lru_cache


class Preprocessor:
    lemmatizer = WordNetLemmatizer()
    stop_word = sw.words('english')

    @staticmethod
    def tokenize(doc):
        list_lemma = []

        try:
            sent_tokenize(doc)
        except Exception as e:
            print ("cant tokenize : " + doc + str(e))
            return

        for sent in sent_tokenize(doc):

            token_iter = iter(pos_tag(wordpunct_tokenize(sent)))

            for token, tag in token_iter:

                # ignore hidden name
                if token[0] == '@':
                    next(token_iter)

                # clean up some token
                token = token.lower()
                token = token.strip()
                token = token.strip('_')
                token = token.strip('*')

                try:
                    token = token.encode('utf-8')
                except Exception:
                    continue

                # If stopword, ignore token and continue
                if token in set(Preprocessor.stop_word):
                    continue

                # If punctuation, ignore token and continue
                if all(char in set(string.punctuation) for char in token):
                    continue


                # Lemmatize token
                try:
                    lemmatized_token = Preprocessor.__lemmatize(token, tag)
                except Exception as e:
                    continue

                list_lemma.append(lemmatized_token)

            list_lemma.append(".")

        return " ".join(list_lemma)

    @staticmethod
    def __lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return Preprocessor.lemmatizer.lemmatize(token, tag)


def select_based_id(data_input, id):
    # new_x = data.filter(['essay_id','essay_set','essay', ''], axis=1)
    return (data_input.loc[data_input["essay_set"] == id])


# AutoGen file data
# Xoa data trong tokenized_data truoc khi train lai, neu khong la no de len

if __name__ == "__main__":

    print ('begin')

    t = time();
    print('Tokenizing : It works good but it goes slow sometimes but its a very good phone I love it')

    print(Preprocessor.tokenize('It works good but it goes slow sometimes but its a very good phones I love it'))
    print(time() - t)
    t = time ()

    with open('data/training_set_rel3.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        count = 0
        t = time()

        tuples = []

        for row in spamreader:

            count += 1
            # Preprocessor.tokenize(row[4])
            row[2] = Preprocessor.tokenize(row[2])

            tuples.append(row);

            if count%200 == 0:
                print(tuples)
                print ("current" + str(count))
                print ("time to process last 200 : " + str(time() - t))
                with open('data/tokenized_data.csv', 'a') as f:
                    cw = csv.writer(f, lineterminator='\n')
                    cw.writerows(r + [""] for r in tuples)
                tuples = []
                t = time()