# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:38:22 2014

@author: hussam
"""


import utility as ut
from math import *
import sent_analysis as twzscore2


def loadzscore(corpus, path):
    dict_3 = [dict(), dict(), dict()]
    f = [0, 0, 0]
    if corpus=='tw':
        zdic = ["data/zneg.txt","data/znet.txt","data/zpos.txt"]
    elif corpus=='txt':
        zdic = ["data/cr0.txt","data/cr1.txt","data/cr2.txt"]
    f[0] = open(path + zdic[0], "r").readlines()
    f[1] = open(path + zdic[1], "r").readlines()
    f[2] = open(path + zdic[2], "r").readlines()
    for i in range(3):
        for line in f[i]:
            line = line.replace("\n", "").split("\t")
            dict_3[i][line[0]] = (float(line[1]))
    return dict_3

def getrealZscore(data, labels):
    #TODO Add an option to construct real Zscore for text.
    # Normally we should reconstruct cr0, cr1, cr2 in a training phase (not write
    # in hard in the repository)

    allvocab = dict()
    dict_3 = [dict(), dict(), dict()]
    count_3 = [0, 0, 0]
    nvocab = 0
    for i in range(len(data)):
        sentence = data[i]
        polarity = labels[i]
        sentence = twzscore2.splitfun4tweet(sentence)
        for token in sentence:
            if (len(token) > 1):
                if token in dict_3[polarity]:
                    dict_3[polarity][token] += 1
                    count_3[polarity] += 1
                else:
                    dict_3[polarity][token] = 1
                    count_3[polarity] += 1
                if allvocab.has_key(token):
                    allvocab[token] += 1
                    nvocab += 1
                else:
                    allvocab[token] = 1
                    nvocab += 1
    all = ["", "", ""]
    xy = [[[], []], [[], []], [[], []]]
    index = 0
    for k, v in allvocab.iteritems():
        for i in range(3):
            if dict_3[i].has_key(k):
                p = ((allvocab[k]) / float(nvocab))
                all[i] = str(k) + "\t" + str((dict_3[i][k] - count_3[i] * p) / (sqrt(count_3[i] * p * (1 - p)))) + "\n" + all[i]
                xy[i][0].append(index)
                xy[i][1].append((dict_3[i][k] - count_3[i] * p) / (sqrt(count_3[i] * p * (1 - p))))
                index += 1
            else:
                p = ((allvocab[k] * 1.0) / nvocab)
                all[i] = str(k) + "\t" + str((0 - count_3[i] * p) / (sqrt(count_3[i] * p * (1 - p)))) + "\n" + all[i]
                xy[i][0].append(index)
                xy[i][1].append((0 - count_3[i] * p) / (sqrt(count_3[i] * p * (1 - p))))
                index += 1
    f0 = open(path + "data/zneg.txt", "w")
    f1 = open(path + "data/znet.txt", "w")
    f2 = open(path + "data/zpos.txt", "w")
    f0.writelines(all[0])
    f1.writelines(all[1])
    f2.writelines(all[2])
    f0.close()
    f1.close()
    f2.close()

if __name__ == '__main__':
    filename = "./corpus/twitter-train-cleansed-B.txt"
    data, labels = zscore2.readFile(filename)
    getrealZscore(data, labels)
