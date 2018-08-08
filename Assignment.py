import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams

sample = pd.read_csv("SMSSpamCollection.tsv",sep='\t',header=None)
sample.columns = ['label', 'body-text']

sample['body-lower'] = sample['body-text'].str.lower()
sample['body-removed-punct'] = sample['body-lower'].apply(lambda x: ''.join([word for word in x if word not in (string.punctuation)]))
sample['body-removed-stopwords'] = sample['body-removed-punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
sample['body-word'] = sample['body-removed-stopwords'].apply(lambda x: RegexpTokenizer("[\w']+").tokenize(x))
sample['body-word-lemmatized'] = sample['body-word'].apply(lambda x: [WordNetLemmatizer().lemmatize(y) for y in x])
sample['body-bigrams'] = sample['body-word-lemmatized'].apply(lambda x: list(ngrams(x, 2)))

bigram_lists = sample.groupby('label').agg({'body-bigrams': 'sum'})
unigram_lists = sample.groupby('label').agg({'body-word-lemmatized': 'sum'})

spam_unigrams = {}
ham_unigrams = {}
spam_bigrams = {}
ham_bigrams = {}

for unigram in unigram_lists.iat[1, 0]:
    if unigram not in spam_unigrams:
        spam_unigrams[unigram] = 1
    else:
        spam_unigrams[unigram] += 1

for unigram in unigram_lists.iat[0, 0]:
    if unigram not in ham_unigrams:
        ham_unigrams[unigram] = 1
    else:
        ham_unigrams[unigram] += 1

for bigram in bigram_lists.iat[1, 0]:
    if bigram not in spam_bigrams:
        spam_bigrams[bigram] = 1
    else:
        spam_bigrams[bigram] += 1

for bigram in bigram_lists.iat[0, 0]:
    if bigram not in ham_bigrams:
        ham_bigrams[bigram] = 1
    else:
        ham_bigrams[bigram] += 1

message = input("Enter the message: ")

message_lower = message.lower()
message_sub = re.sub(r'[^\w\s]', '', message_lower)
message_word = RegexpTokenizer("[\w']+").tokenize(message_sub)
message_removed_stopwords = [word for word in message_word if not word in set(stopwords.words('english'))]
message_lemmatized_word = []
for word in message_removed_stopwords:
    message_lemmatized_word.append(WordNetLemmatizer().lemmatize(word))

bigram_list = list(ngrams(message_lemmatized_word, 2))

V_ham = len(ham_unigrams)
V_spam = len(spam_unigrams)

def cal_bigram_ham_prob(w1, w2):
    try:
        N = ham_unigrams[w1]
    except KeyError:
        N = 0

    try:
        C = ham_bigrams[w1, w2]
    except KeyError:
        C = 0

    prob = (C + 1) / (N + V_ham)
    return prob

def cal_bigram_spam_prob(w1, w2):
    try:
        N = spam_unigrams[w1]
    except KeyError:
        N = 0

    try:
        C = spam_bigrams[w1, w2]
    except KeyError:
        C = 0

    prob = (C + 1) / (N + V_spam)
    return prob

prob_ham = 1
prob_spam = 1

for bigram in bigram_list:
    prob_ham = prob_ham * cal_bigram_ham_prob(bigram[0], bigram[1])
    prob_spam = prob_spam * cal_bigram_spam_prob(bigram[0], bigram[1])

print(prob_ham)
print(prob_spam)


if(prob_ham > prob_spam):
    print(message + " => ham")
else:
    print(message + " => spam")