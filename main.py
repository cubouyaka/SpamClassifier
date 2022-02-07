import my_functions


file_name = 'SMSSpamCollection'
file_train_name = 'train'
file_test_name = 'test'

seed_split = 1234
proba_split = 0.6

# Split our DataSet between a training and testing set
my_functions.split_lines(file_name, seed_split, file_train_name, file_test_name, proba_split)

(spam_ratio, words, spamicity) = my_functions.naive_bayes_train(file_train_name)


def classify_spam(sms):
	"""Returns True if the message 'sms' is predicted to be a spam."""
	return my_functions.naive_bayes_predict(spam_ratio, words, spamicity, sms) > 0.5


(recall, precision) = my_functions.naive_bayes_eval(file_test_name,lambda x:classify_spam(x))

#Let's print some results
print("Results of the evaluation of our classifier:")
print("Recall = "+str(round(recall*100,2))+"%")
print("Precision = "+str(round(precision*100,2))+"%")