# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 01:04:18 2014

@author: hussam
"""
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords


def stopwords():
    stht = {}
    stopwords = open("data/frstopwords.txt", 'r').readlines()
    for word in stopwords:
        word = word.replace("\n", " ").lower()
        if not (word in stht):
            stht[word.split(" ")[0]] = 1
    return stht


def standardstopwords():
    return set(stopwords.words('english'))


def setmword(word):
    return PorterStemmer().stem_word(word)


def printFmeasur2(conf_mat):
    for i in range(2):
        print "%d   %d" % (conf_mat[i][0], conf_mat[i][1])
    # print conf_mat[0][0]+conf_mat[1][0]+conf_mat[2][0]
    pneg = conf_mat[0][0] * 1.0 / (conf_mat[0][0] + conf_mat[1][0])
    pneut = conf_mat[1][1] * 1.0 / (conf_mat[0][1] + conf_mat[1][1])

    rneg = conf_mat[0][0] * 1.0 / (conf_mat[0][0] + conf_mat[0][1])
    rneut = conf_mat[1][1] * 1.0 / (conf_mat[1][0] + conf_mat[1][1])

    f0 = 2 * (pneg * rneg) / (pneg + rneg)
    f1 = 2 * (pneut * rneut) / (pneut + rneut)

    # print "PN:%s RN:%s\nPNN:%s RNN:%s\nPP:%s
    # RP:%s"%(pneg,rneg,pneut,rneut,ppos,rpos)
    print "(FN:%s  FNN:%s   FP:%s )" % (round(f0, 3), round(f1, 3), round((f0 + f1) / 2, 3))


def printFmeasur(conf_mat):
    # for i in range(3):
    #    print "%d   %d   %d"%(conf_mat[i][0],conf_mat[i][1],conf_mat[i][2])
    # print conf_mat[0][0]+conf_mat[1][0]+conf_mat[2][0]
    pneg = conf_mat[0][0] * 1.0 / \
        (conf_mat[0][0] + conf_mat[1][0] + conf_mat[2][0])
    pneut = conf_mat[1][1] * 1.0 / \
        (conf_mat[0][1] + conf_mat[1][1] + conf_mat[2][1])
    ppos = conf_mat[2][2] * 1.0 / \
        (conf_mat[0][2] + conf_mat[1][2] + conf_mat[2][2])
    rneg = conf_mat[0][0] * 1.0 / \
        (conf_mat[0][0] + conf_mat[0][1] + conf_mat[0][2])
    rneut = conf_mat[1][1] * 1.0 / \
        (conf_mat[1][0] + conf_mat[1][1] + conf_mat[1][2])
    rpos = conf_mat[2][2] * 1.0 / \
        (conf_mat[2][0] + conf_mat[2][1] + conf_mat[2][2])
    f0 = 2 * (pneg * rneg) / (pneg + rneg)
    f1 = 2 * (pneut * rneut) / (pneut + rneut)
    f2 = 2 * (ppos * rpos) / (ppos + rpos)
    # print "PN:%s RN:%s\nPNN:%s RNN:%s\nPP:%s
    # RP:%s"%(pneg,rneg,pneut,rneut,ppos,rpos)
    print "(FN:%s  FNN:%s   FP:%s M:%s M2:%s)" % (round(f0, 2), round(f1, 2), round(f2, 2), round((f0 + f1 + f2) / 3, 2), (f0 + f2) / 2)


def loadBleiVocabFileIntoList(vocabfile):
    f = open(vocabfile, 'r').readlines()
    ht = []
    for line in f:
        line = line.replace("\n", "")
        token = line.split("[")[0]
        ht.append(token)
    return ht


def SpareArticles(sentence):
    news = ""
    for token in sentence:
        if token in ["!", ";", ",", "?", ".", "/"]:
            news += " " + token
        else:
            news += token
    return news


def getPosTags(sentence):
    text = nltk.word_tokenize(sentence)
    return nltk.pos_tag(text)
