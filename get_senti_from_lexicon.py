# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:29:32 2014

@author: hussam
"""
from math import fabs


def loadswn(path="lexicon/"):
    f = open(path + "SentiWordNet_3.0.0_20130122.txt", 'r')
    lines = f.readlines()
    lexdic = dict()
    lexposdict = dict()
    i = 0
    for line in lines:
        i += 1
        l = line.replace("\n", "").split("\t")
        if float(l[2]) - float(l[3]) > 0.25:
            pol = 1
        elif float(l[3]) - float(l[2]):
            pol = -1
        else:
            pol = 0
        lexdic[l[4].split("#")[0]] = pol
        lexposdict[l[4].split("#")[0] + "#" + l[0]] = pol
    return lexdic, lexposdict


def loadLexicon(path="lexicon/"):
    f = open(path + "subjclueslen1-HLTEMNLP05.tff", 'r')
    lines = f.readlines()
    lexdic = dict()
    for line in lines:
        l = line.replace("\n", "").split(" ")
        word = ""
        for item in l:
            it = item.split("=")
            if len(it) < 2:
                continue
            if it[0] == "word1":
                word = it[1]
            if it[0] == "priorpolarity":
                if it[1] == "negative":
                    lexdic[word] = -1
                elif it[1] == "positive":
                    lexdic[word] = 1
                else:
                    lexdic[word] = 0
    return lexdic


def loadLU(path="lexicon/"):
    f = open(path + "negative-words.txt", 'r')
    lines = f.readlines()
    ludic = dict()
    for line in lines:
        l = line.replace("\n", "")
        ludic[l] = -1
    f = open(path + "positive-words.txt", 'r')
    lines = f.readlines()
    for line in lines:
        l = line.replace("\n", "")
        ludic[l] = 1
    return ludic


def getWordsenti(word, ludic):
    if word in ludic:
        return ludic[word]
    else:
        return 2  # not exist
