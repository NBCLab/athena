# -*- coding: utf-8 -*-
"""
Created on Thu May  5 21:58:48 2016

Number words.

@author: salo
"""
import re
import itertools
import nltk
from itertools import groupby
from operator import itemgetter
from word2number import w2n


def consec(data):
    ranges = []
    for key, group in groupby(enumerate(data), lambda (index, item): index - item):
        group = map(itemgetter(1), group)
        ranges.append(range(group[0], group[-1]+1))
        
    return ranges


def convert_words_to_numbers(sentence):
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
               "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
               "zero", "ten", "eleven", "twelve", "thirteen", "fourteen",
               "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
               "hundred", "thousand", "million", "billion", "trillion"]
    
    string = "|".join(numbers)
    full = "((" + string + ").+(" + string + "))|(" + string + ")"
    bb = re.compile(full, re.IGNORECASE)
    
    sentence2 = re.sub(r"\band\b", "", sentence)
    sentence2 = sentence2.replace("-", " ").replace(",", "")
    sentence2 = re.sub(r"\s+", " ", sentence2)
    if any(number in nltk.word_tokenize(sentence2.lower()) for number in numbers):
        found = re.search(bb, sentence2).group()
        words = nltk.word_tokenize(found)
        
        idx = [i for i in range(len(words)) if words[i].lower() in numbers]
        idx2 = consec(idx)
        groups = [[words[i] for i in group] for group in idx2]
        nums = [str(w2n.word_to_num(" ".join(group))) for group in groups]
        
        for i in range(len(groups)):
            sentence = sentence[:sentence.index(groups[i][0])] + nums[i] + sentence[sentence.index(groups[i][-1])+len(groups[i][-1]):]
    return sentence


def find_candidates(sentences):
    candidate_terms = ["subjects", "users", "patients", "men", "women", "male", "female", "controls"]
    or_term = "|".join(candidate_terms)
    search_term = re.compile(r"\b(%s)\b" % or_term, re.IGNORECASE)
    
    out_sentences = []
    for sentence in sentences:
        if re.search(search_term, sentence):
            out_sentences.append(sentence)
    return out_sentences


def reduce_candidates(sentences):
    out_sentences = []
    for sentence in sentences:
        if any(char.isdigit() for char in sentence):
            out_sentences.append(sentence)
    return out_sentences

