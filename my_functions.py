import random
import numpy as np

def split_lines(input_file, seed, output1, output2, proba):
	"""Distributes the lines of 'input_file' to 'output1' and 'output2' randomly,
	with a probability 'proba' to go to 'output1'
	Args:
		input_file: a string, the name of the input file.
		seed: an integer, the seed of the pseudo-random generator used. The split
			will be different with different seeds. Conversely, using the same
			seed and the same input will give exactly the same outputs.
		output1: a string, the name of the first output file.
		output2: a string, the name of the second output file.
	"""

	try:
		input_file = open(input_file, 'r')
		output1_file = open(output1, 'w+')
		output2_file = open(output2, 'w+')

		random.seed(seed)

		for line in input_file.readlines():
			if random.random() < proba :
				output1_file.write(line)
			else:
				output2_file.write(line)

		input_file.close()
		output1_file.close()
		output2_file.close()

	except FileNotFoundError:
		print ("The file " + input_file+" does not exists")


def tokenize_and_split(sms_file):
	"""Parses and tokenizes the sms data, splitting 'spam' and 'ham' messages.
	Args:
		sms_file: a string, the name of the input SMS data file.
	Returns:
		A triple (words, spams, hams):
		- words is a dictionary mapping each word to a unique, dense 'word index'.
		- spams is a list of the 'spam' messages, encoded as lists of word indices.
		- hams is like spams, but for 'ham' messages.
	"""

	input_file = open(sms_file,'r')
	words = {}
	spams , hams = [] ,[]
	index_words = 0

	for line in input_file.readlines():
		mode = "none"
		tmp = []
		for word in line.split():
			if mode == "none":
				mode = word
			else:
				if word not in words:
					words[word] = index_words
					index_words += 1
				tmp.append(words[word])

		if mode == "spam":
			spams.append(tmp)
		elif mode == "ham":
			hams.append(tmp)

	return tuple((words,spams,hams))


def compute_frequencies(num_words, documents):
	"""Computes the frequency of words in a corpus of documents.
	Args:
		num_words: the number of words that exist.
		documents: a list of lists of integers.
	Returns:
		A list of floats of length num_words: element #i will be the ratio
		(in [0..1]) of documents containing i, i.e. the ratio of indices j
		such that "i in documents[j]".
	"""

	ratios = [0.0] * num_words
	for document in documents:
		for word in set(document):
			ratios[word] += 1
	return [ ratio / len(documents) for ratio in ratios ]


def naive_bayes_train(sms_file):
	"""Performs the "training" phase of the Naive Bayes estimator.
	Args:
		sms_file: a string, the name of the input SMS data file.
	Returns:
		A triple (spam_ratio, words, spamicity) where:
		- spam_ratio is a float in [0..1] and is the ratio of SMS marked as 'spam'.
		- words is the dictionary output by tokenize_and_split().
		- spamicity is a list of num_words floats, where num_words = len(words) and
			spamicity[i] = (ratio of spams containing word #i (across all spams)) /
			(ratio of SMS (spams and hams) containing word #i (across all SMS))
	"""

	input_file = open(sms_file, 'r')
	(words, spams, hams) = tokenize_and_split(sms_file)
	num_words = len(words)

	spamicity = compute_frequencies(num_words,spams)
	sms = compute_frequencies(num_words,hams+spams)

	spamicity = [spamicity[i] / sms[i] for i in range(num_words)]

	return (len(spams)/(len(spams)+len(hams)), words, spamicity)


def naive_bayes_predict(spam_ratio, words, spamicity, sms):
	"""Performs the "prediction" phase of the Naive Bayes estimator.
	Args:
		spam_ratio: see output of naive_bayes_train
		words: see output of naive_bayes_train
		spamicity: see output of naive_bayes_train
		sms: a string
	Returns:
		The estimated probability that the given sms is a spam.
	"""

	proba = 1.0

	for word in set(sms.split()):
		if word in words:
			proba *= spamicity[words[word]]

	return proba*spam_ratio


def msg_text(msg, words):
	"""Transform a message encoded as indexes to a string 
	Args:
		msg: a list of indexes in words representing a sms
		words: the dictionnary containing the words with their index
	Returns:
		A string: the message as a string

	"""
	invert_words = dict((v, k) for k, v in words.items())
	result = ""
	for index in msg:
		result += invert_words[index]+" "

	return result


def naive_bayes_eval(test_sms_file, f):
	"""Evaluates a spam classifier.
	Args:
		test_sms_file: a string, the name of the input 'test' SMS data file.
		f: a function. f(sms), where sms is a string,should return 1 if sms 
		is classified as spam, and 0 otherwise.
	Returns:
		A pair of floats (recall, precision): 'recall' is the ratio (in [0,1]) of
		spams in the test file that were successfully identified as spam, and
		'Precision' is the ratio, among all sms that were predicted as spam by f, of
		sms that were indeed spam.
	"""

	(spam_ratio, words, spamicity) = naive_bayes_train(test_sms_file)
	(words, spams, hams) = tokenize_and_split(test_sms_file)

	recall, nb_prediction, nb_success = 0.0, 0.0, 0.0

	for msg in spams+hams:
		if f(msg_text(msg,words)):
			nb_prediction += 1
			if msg in spams:
				recall += 1
				nb_success += 1


	recall = recall / len(spams)
	precision = nb_success / nb_prediction if nb_prediction != 0 else 1.0

	return tuple((recall, precision))




