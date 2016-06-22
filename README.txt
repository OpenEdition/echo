
** I/O Files **

INPUT files (file to evaluate) are ALREADY in ./input/ 
	Nevertheless, you can modify directly this files. 
	Or you can use interactive python shell and use the function getResult() (line 306 in sent_analysis.py) and give as input a list of strings.
	Currently, the interactive python shell implement only text (not tweet).


OUTPUT files
	- a new eval/hxt*.txt is created for tweet (according * with the option)
	- a new eval/review-z.txt is created for text	


----------------------------

** Running **


TAKE CARE: FOR THE FIRST USE TRAINING IS MANDATORY FOR TWEET.

Echo offer a sentiment analysis on tweet and on text (specially text coming from shs article review)

For start sentiment anlysis in command line, running on linux console:
$python sent_analysis.py
	-Choose which type of corpus do you want to use (text ot tweet)
	-Choose training or not (mandatory if you run for the first time Echo)
	-Define process's type to run sentiment anlaysis.


----------------------------

** Training **

If you choose tweet option:
 	- For the first time, you need to training for tweet (at least once for each option) then you could processing sentiment analysis at any time.
	  Training will create your differents files model:
	  3 options are avalaible for training :z-score, polarity and twitter Dictionnary ; POS option is not working properly currently.
	
If you choose text option:
	- training is include. You don't need to train, you process sentiment analysis directly.


----------------------------

** Analysing **

	- Tweet, 3 option, process is avalaible (Z-score process ; Polarity ; Twitter)
	- Text, zscore option is choosen by default. There is no other option for text.
It could take a while, specialy if the input file is important.

RETURN: Name of the ouptut/file.


----------------------------

** Evaluation **

Evaluation is automatically done after analysing.
	- Tweet, evaluation is done by control refence (see perl programm eval/score-semeval2014-task9-subtaskB.pl)
	- Text, evaluation is done by cross-validation

Evaluation is displayed at the end of the process.


----------------------------

** In an interactive shell (only for txt option) **

You must have PROCESS AT LEAST ONCE in cli the text option (to create different configuration file)
In your virtualenv:
$python
$import import sent_analysis
$my_list=[u'mysentence0', u'mysentencei', ....]
$sent_analysis.getResult(my_list)
OUTPUT : list= [(opinioni, sentencei)]


----------------------------

** Using the Command Line Interface (CLI) **

You can access all details by typing :
```
python sent_analysis.py -h
```
This will give you the following informations :
optional arguments:
  -h, --help            show this help message and exit
  -train, --train
                        train file path
  -c, --corpus
                        modes; "txt" for text or "tw" for tweets)
  -test, --test
                        test file path
  -f, --feature
                        type of features : "zs" for z-score, "pol" for
                        polarity or "dic" for twitterDictionary
  -t, --trainingFlag    use this flag to enable training
  -o, --output
                        fichier de sortie


----------------------------

** About the different files **

Some ressources and input-data files are uses uses during process:
	./lexicon:	-> Usefull only for POLARITY OPTION):
		-negative-words.txt:		Dictionnary of negative words for tweet. 
		-positive-words.txt: 		Dictionnary of positive words for tweet. 
		-subjclueslen1-HLTEMNLP05.tff:	The Subjectivity Lexicon (list of subjectivity clues) that is part of OpinionFinder
		-SentiWordNet_3.0.0_20130122.txt SentiWordNet is a lexical resource for opinion mining (http://sentiwordnet.isti.cnr.it/)
	./data:
		-twitterDic.txt is a lexicon of expression and emotion icons (only for method twitter Dictionnary)
		-frstopŵords.txt is a corpus uses in pre-process for tokenize text. It is used with nltk library. (To fit the input).
		-cr*.txt is a zscore reference construct for TEXT during the training phase. Normally we reconstruct this files each times we launched training proccess (not implemented yet).
		-z*.txt is a zscore reference construct for TWEET during the training phase. Normally we reconstruct this files each times we launched training proccess (not implemented yet).
	
	./input:
		-semeval-tweet-test-B-input.txt is an input file for tweet (for example).
		-review-input.txt is a an input file for review text. The opinion tagged are present but not used for evaluation. 

	./eval:		-> used to evaluate the result of sentiment analysing on tweet:
		-semeval-tweet-test-B-reference.txt (text reference to marked tweet sentiment analyse)
		-score-semeval2014-task9-subtaskB.pl (perl code to evaluate the result)

	./corpus:
		-review.txt is a review's annotated hand-made corpus
		-twitter-train-cleansed-B.txt is a tweet's annotated hand-made corpus

