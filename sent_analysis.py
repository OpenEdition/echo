# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:33:02 2014

@author: Hussam Hamdan

Modified version by Gaël Guibon (command line interface added)
"""
import re
import happy_tokeniser as htok
import z_score as zscore
from collections import OrderedDict
import utility as ut
import get_senti_from_lexicon as sentii
from sklearn.externals import joblib
from sklearn import svm, cross_validation
from sklearn.datasets import load_svmlight_file
tokeniser = htok.Tokenizer(preserve_case=False)
import shutil
import tempfile
import os
import json
import argparse

# argparse added by Gael Guibon
parser = argparse.ArgumentParser(description='echo by Hussam Hamdam. Forked by Gaël Guibon in order to add a CLI \n Sentiment analysis classifier by polarity.')
parser.add_argument('-train','--train', metavar='TRAIN', type=str, help='train file path')
parser.add_argument('-c','--corpus', metavar='MODES', type=str, help='modes; "txt" for text or "tw" for tweets)')
parser.add_argument('-test','--test', metavar='TEST', type=str, help='test file path')
parser.add_argument('-f','--feature', metavar='FEATS', type=str, help='type of features : "zs" for z-score, "pol" for polarity or "dic" for twitterDictionary')
parser.add_argument('-t', '--trainingFlag', action='store_true', help='use this flag to enable training')
parser.add_argument('-o','--output', metavar='OUTPUT', type=str, help='output file path')

args = parser.parse_args()



def readFile(filename):
    content = open(filename, "r").readlines()
    data = []
    labels = []
    i = 0
    count = [0, 0, 0]
    for line in content:
        i += 1
        #line=unicodedata.normalize('NFKD', line).encode('ascii','ignore')
        line = line.split("\t")
        line[3] = line[3].replace("\n", "")
        line[2] = 2 if line[2] == "positive" else 0 if line[2] == "negative" else 1
        count[line[2]] += 1
        data.append(line[3])
        labels.append(line[2])
    return data, labels

def loadDic(path, corpus):
    if corpus=='tw':
        f = open(path + "data/twitterDic.txt", "r")
    elif corpus=='txt':
        f = open(path + "data/twitterDic.txt", "r")
    dicth = dict()
    lines = f.readlines()
    for line in lines:
        if line.find(" = ") != -1:
            line = line.split(" =")
            dicth[line[0].lower()] = line[1].replace("\n", "").lower()
    return dicth

def getpolarityid(polarity):
    if polarity == "negative":
        return 0
    elif polarity == "neutral":
        return 1
    elif polarity == "positive":
        return 2

def getPolaritynam(polarity):
    if polarity == 0:
        return "negative"
    elif polarity == 1:
        return "neutral"
    elif polarity == 2:
        return "positive"

def splitfun4tweet(doc):
    doc = doc.decode('utf8', errors='ignore').encode('ascii', errors='ignore')
    doc = doc.replace("\t", " ").replace("\r", "").replace("\n", "").lower()
    doc = doc.replace("\\u002c", ",").replace("\\u2019", "'")
    doc = doc.replace("\\u2013", "-").replace("\\u2013", "-")
    y = re.findall(r'@[a-z0-9]+', doc)
    for y1 in y:
        doc = doc.replace(y1, "uuser")
    y = re.findall(r'http://[a-z0-9./]+', doc)
    for y1 in y:
        doc = doc.replace(y1, "http")
    res = tokeniser.tokenize(doc)
    if "\u2013" in res:
        print res
    if "\\u2013" in res:
        print res
        print "h"
    bires = []
    for i in range(len(res)-1):
        bires.append(res[i] + " " + res[i + 1])
    return res + bires

def predictzvalues(text, zdict_3, threshold=3):
    sumz = [0, 0, 0, 0]
    text = splitfun4tweet(text)
    for token in text:
        if (len(token) > 1):
            for i in range(len(zdict_3)):
                if ((token in zdict_3[0]) and zdict_3[i][token]>=threshold):
                    sumz[i] += 1
    return sumz

def getTrainingFile(data, labels, vocabfile, textfile, prepolarity, Z_score, POS, TwitterDict, corpus, tokenlngth=1, zthreshold=3, path='./'):
    '''This function take the data and their labels with some options in order to construct a training file
    data: the orginal textual tweets
    labels: the polarity of each tweet
    vocabfile: in which the vocabulary is indexed and saved
    textfile: the output training file
    prepolarity,Z_score,POS,TwitterDict : if we want using the prepolarity, z score, pos, twitter dictionary features.
    it is not recommended to use POS.'''
    f = open(textfile, "w")
    vocf = open(vocabfile, "w")
    vocdict = dict()
    i = 0
    if prepolarity:
        vocdict["positive1"] = i; i+=1
        vocdict["negative1"] = i; i+=1
        vocdict["neutral1"] = i; i+=1
        lexdic = sentii.loadLexicon()
        ludic = sentii.loadLU()
    if Z_score:
        vocdict["zscore1"] = i; i+=1
        vocdict["zscore2"] = i; i+=1
        vocdict["zscore3"] = i; i+=1
        vocdict["allzscore"] = i; i+=1
        z_dict = zscore.loadzscore(corpus, path)
    if POS:
        vocdict["pos0"] = i; i+=1
        vocdict["pos1"] = i; i+=1
        vocdict["pos2"] = i; i+=1
        vocdict["pos3"] = i; i+=1
        vocdict["pos4"] = i; i+=1

    # tweet dictionary loading (expression,emotion icons)
    if TwitterDict:
        dicht = loadDic(path, corpus)
    vocid = i

    for i in range(len(data)):
        text = data[i]
        polarity = labels[i]
        if POS:
            pos = ut.getPosTags(text)
            postags = [0, 0, 0, 0, 0]
            paspect = str(pos)
            postags[0] = len(paspect.split("'NN")) - 1
            postags[1] = len(paspect.split("'JJ")) - 1
            postags[2] = len(paspect.split("'RB'")) - 1
            postags[3] = len(paspect.split("'VB")) - 1
            postags[4] = len(paspect.split("'CC'")) - 1
        if Z_score:
            zsum = predictzvalues(text, z_dict, zthreshold)
        # add the emotions
        if TwitterDict:
            for k, v in dicht.iteritems():
                if text.find(k) != -1:
                    text += " " + v
        newtxt = splitfun4tweet(text)
        instance = ""
        sentencedict = dict()
        priorpol = [0, 0, 0, 0]
        for tok in newtxt:
            if len(tok) > tokenlngth:
                if prepolarity:
                    pol1 = sentii.getWordsenti(tok, ludic)
                    pol2 = sentii.getWordsenti(tok, lexdic)
                    priorpol[pol2 + 1] += 1  # 61 +1 alone
                    priorpol[pol1 + 1] += 1
                index = -1
                if vocdict.has_key(tok):
                    index = vocdict[tok]
                else:
                    vocdict[tok] = vocid
                    index = vocid
                    vocid += 1
                if sentencedict.has_key(index):
                    sentencedict[index] += 1
                else:
                    sentencedict[index] = 1
        i = 0
        instance = ""
        if prepolarity:
            instance += " %d:%d %d:%d %d:%d" % (i, priorpol[0], i+1, priorpol[1], i+2, priorpol[2])
            i = i + 3
        if Z_score:
            instance += " %d:%d %d:%d %d:%d %d:%d" % (i, zsum[0], i+1, zsum[1], i+2, zsum[2], i+3, zsum.index(max(zsum)))
            i = i + 4
        if POS:
            instance += " %d:%d %d:%d %d:%d %d:%d %d:%d" % (i, postags[0], i+1, postags[1], i+2, postags[2], i+3, postags[3], i+4, postags[4])
            i = i + 5
        d_sorted_by_value = OrderedDict(sorted(sentencedict.items(), key=lambda x: x[0]))
        for k, v in d_sorted_by_value.iteritems():
            instance += " " + str(k) + ":" + str(v)
        f.write(str(polarity) + instance + "\n")
    line = ""
    d_sorted_by_value = OrderedDict(sorted(vocdict.items(), key=lambda x: x[1]))
    for k, v in d_sorted_by_value.iteritems():
        line += str(k.encode("ascii", "ignore")) + "\t" + str(v) + "\n"
    vocf.write(line)
    vocf.close()
    f.close()

def getTestFile(path, inputfile, outputsvmfile, inputvocab, classifier, prepolarity, Z_score, POS, TwitterDict, corpus, zthreshold=3, tokenlngth=1):
    '''This function generates the test file, it takes the test file path and the vocabulary file (inputvocab), the classifier (classifier) and the options
    in order to predict the polarity of each tweet in the test file'''

    if prepolarity:
        lexdic = sentii.loadLexicon()
        ludic = sentii.loadLU()
        lexswn, lexposswn = sentii.loadswn()
    if Z_score:
        z_dict = zscore.loadzscore(corpus, path)
    resfile = open(outputsvmfile, "w")
    if TwitterDict:
        dicht = loadDic(path, corpus)
    index = 0
    lines = open(inputfile, "r")
    all_txt = ""
    for row1 in lines:
        row1 = row1.replace("\n", "").split("\t")
        id1 = row1[0]
        id2 = row1[1]
        sentence = row1[3].lower()
        zsum = [0, 0, 0, 0]
        if Z_score:
            zsum = predictzvalues(sentence, z_dict, zthreshold)
        rowhash = {}
        count = 0
        if TwitterDict:
            for k, v in dicht.iteritems():
                if sentence.find(k) != -1:
                    sentence += " " + v
        newt = splitfun4tweet(sentence)
        priorpol = [0, 0, 0, 0]
        postags = [0, 0, 0, 0, 0]
        if POS:
            aspect = ut.getPosTags(sentence)
            paspect = str(aspect)
            postags[0] = len(paspect.split("'NN")) - 1
            postags[1] = len(paspect.split("'JJ")) - 1
            postags[2] = len(paspect.split("'RB'")) - 1
            postags[3] = len(paspect.split("'VB")) - 1
            postags[4] = len(paspect.split("'CC'")) - 1
        for token in newt:
            if (len(token) > tokenlngth):
                if prepolarity:
                    pol1 = sentii.getWordsenti(token, ludic)
                    pol2 = sentii.getWordsenti(token, lexdic)
                    priorpol[pol2 + 1] += 1  # 61 +1 alone
                    priorpol[pol1 + 1] += 1  # alone 61 +1
                count += 1
                if inputvocab.has_key(token):
                    index = inputvocab[token]
                # filling the hashtable for each file index:number of occurence
                    if rowhash.has_key(index):
                        rowhash[index] += 1
                    else:
                        rowhash[index] = 1
        x_test = [0 for i in range(len(inputvocab))]
        d_sorted_by_value = OrderedDict(sorted(rowhash.items(), key=lambda x: x[0]))
        for k, v in d_sorted_by_value.iteritems():
            x_test[k] = v
        i = 0
        if prepolarity:
            x_test[i], x_test[i+1], x_test[i+2] = (priorpol[0], priorpol[1], priorpol[2])
            i = i + 3
        if Z_score:
            x_test[i], x_test[i+1], x_test[i+2], x_test[i+3] = zsum[0], zsum[1], zsum[2], zsum.index(max(zsum))
            i = i + 4
        if POS:
            x_test[i], x_test[i+1], x_test[i+2], x_test[i+3], x_test[i+4] = postags[0], postags[1], postags[2], postags[3], postags[4]
            i = i + 5
        y_pred = classifier.predict(x_test)
        all_txt += "%s\t%s\t%s\t%s\n" % (id1, id2, getPolaritynam(y_pred[0]), row1[3])
    resfile.write(all_txt)
    resfile.close()

def getFileName(prepolarity, z_score, POS, Twittdic):
    fn = "t"
    if z_score:
        fn += "-z"
    if prepolarity:
        fn += "-pol"
    if POS:
        fn += "-pos"
    if Twittdic:
        fn += "-dic"
    return fn


def loadVocbFile(vocfile):
    hasht = dict()
    f = open(vocfile, "r")
    lines = f.readlines()
    i = 0
    for line in lines:
        voc = line.split("\t")
        hasht[voc[0]] = i
        i += 1
    f.close()
    return hasht

def writeInputFile(txt_lst, filename):
    ftxt_lst = [''.join(('NA\t{}\tunknwn\t'.format(txt_lst.index(line)), line, '\n')) for line in txt_lst]
    with open(filename, 'w') as f:
        f.writelines(ftxt_lst)

def getResult(txt_lst, corpus='txt', option='zs', training='N'):
    """Tagged a list of sentence in positive, negative or neutral opinion
    @param : List[string] ; the text (list of sentence) that you want to tagged. Each value of the list is a sentence.
    @return: List[(set)] ; which the set is (sentence, opinion)"""

    inp_dir = tempfile.mkdtemp(prefix = 'data_sent_analysis')
    out_dir = tempfile.mkdtemp(prefix='eval_sent_anlysis')
    inputf = os.path.join(inp_dir, 'tmpdata.txt')
    lst_tmp = [st.encode('utf-8') for st in txt_lst]
    writeInputFile(lst_tmp, inputf)
    outputf = os.path.join(out_dir, 'tmpeval.txt')
    print "Model Loading ..."
    vocabhash = loadVocbFile("./data/reviewvocab.txt")
    classifier1 = joblib.load("./review.pkl")
    print "Predicting ..."
    pol = pos = dic = False
    zs = True
    getTestFile('./', inputf, outputf, vocabhash, classifier1, pol, zs, pos, dic, corpus)
    with open(outputf) as f:
        resultat = []
        for line in f.readlines():
            line_in_list = line.split("\t")
            if (line_in_list[2] and line_in_list[3]) is not None:
                result_by_line = (line_in_list[2].decode('utf-8'), (line_in_list[3].replace('\n', '')).decode('utf-8'))
                print(type(result_by_line))
                resultat.append(result_by_line)
            else:
                pass
    #retour = json.dumps(resultat)
    os.unlink(outputf)
    os.unlink(inputf)
    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
    return resultat

# modified by Gael Guibon 
if __name__ == '__main__':
    zs = dic = pol = pos = False

    if args.corpus=='tw':
        filename="./corpus/twitter-train-cleansed-B.txt"
        svmfname = "data/tweet.txt"
        data, labels = readFile(args.train)    
        x = args.feature
        y = args.training
        if x == "zscore":
            zs = True
        elif x == "polarity":
            pol = True
        elif x == "twitterDictionary":
            dic = True
        else: raise NameError('Invalid Feature Option')

        if y==False:
            print "Model Loading  ..."
            vocabhash = loadVocbFile("data/tweetvocab"+getFileName(pol,zs,pos,dic)+".txt")
            outf = "eval/hx" + getFileName(pol, zs, pos, dic) + ".txt"
            classifier1 = joblib.load(getFileName(pol, zs, pos, dic) + ".pkl")
        elif y==True:
            print "preprocessing ..."
            getTrainingFile(data, labels, "data/tweetvocab" + getFileName(pol,zs,pos,dic) + ".txt", svmfname, pol, zs, pos, dic, corpus)
            vocabhash = loadVocbFile("data/tweetvocab" + getFileName(pol,zs,pos,dic) + ".txt")
            outf = "eval/hx" + getFileName(pol, zs, pos, dic) + ".txt"
            print "training"
            x_train, y_train = load_svmlight_file(svmfname)
            classifier1 = svm.LinearSVC()
            classifier1.fit(x_train, y_train)
            joblib.dump(classifier1, getFileName(pol,zs,pos,dic) + ".pkl")
        else: raise NameError('Invalid Option: training or not training?')

        print "Predicting ..."
        getTestFile('./', "input/semeval-tweet-test-B-input.txt", outf, vocabhash, classifier1, pol, zs, pos, dic, corpus)
        print "Evaluation ..."
        prog = "perl eval/score-semeval2014-task9-subtaskB.pl " + outf
        import subprocess
        p = subprocess.Popen(prog, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
        print(p)

    elif args.corpus=='txt':
        filename = './corpus/review.txt'
        data, labels = readFile(args.train)
        svmfname = "trainab"
        zthreshold=-0.5
        zs = True # For corpus text: only zscore is avalaible
        print "preprocessing ..."
        getTrainingFile(data, labels, "data/reviewvocab.txt", svmfname, pol, zs, pos, dic, args.corpus, zthreshold)
        vocabhash = loadVocbFile("data/reviewvocab.txt")
        outf = "eval/review-z.txt"
        print "Training ..."
        x_train, y_train = load_svmlight_file(svmfname)
        classifier1 = svm.LinearSVC()
        classifier1.fit(x_train, y_train)
        joblib.dump(classifier1, "review.pkl")
        print "predicting"
        getTestFile('./', "input/review-input.txt", outf, vocabhash, classifier1, pol, zs, pos, dic, args.corpus)
        print "Evaluation ..."
        scores = cross_validation.cross_val_score(classifier1, x_train, y_train, cv=5)
        print scores

    else: raise NameError('Invalid Option : Please use "python sent_analysis.py -h" to see all available options')

