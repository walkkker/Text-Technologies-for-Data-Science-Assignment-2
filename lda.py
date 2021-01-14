from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from pprint import pprint


import re
from stemming.porter2 import stem
import math

def tokenization(content):
	word_tokens = re.sub(r'[!@#$%^&*()_+{}|:"<>?,./;\'[\]\-=]+', ' ', content).lower().split()
	return word_tokens

def stop_words(word_tokens, stop_word_read):
	stop_word_tokens = tokenization(stop_word_read)
	filter_stop = [w for w in word_tokens if w not in stop_word_tokens]
	return filter_stop

def porter_stemmer(filter_stop):
	filter_stemmer = [stem(tokens) for tokens in filter_stop]
	return filter_stemmer

def pre_processing(content, stop_word_read):
	word_tokens = tokenization(content)
	filter_stop = stop_words(word_tokens, stop_word_read)
	filter_stemmer = porter_stemmer(filter_stop)
	return filter_stemmer


def lda_model(lda, common_dictionary, corpus_list, num_corpus):
	#Since tuple doesn't support writing operation, we use list so that can write inside values
	overall_score = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],[11,0],[12,0], \
						[13,0], [14,0], [15,0], [16,0], [17,0], [18,0], [19,0]]
	#for corpus we chose to calculate
	corpus = [common_dictionary.doc2bow(text) for text in corpus_list]

	for doc in corpus:
		topic_scores = lda.get_document_topics(doc, minimum_probability=0.00)
		for i in range (0,20):
			if overall_score[i][0]==topic_scores[i][0]:
				overall_score[i][1]+=topic_scores[i][1]

	#in the format [(topicID, average_in_corpus)]
	average_score = [(topic_sum[0],topic_sum[1]/num_corpus) for topic_sum in overall_score]

	#sort the list to get the largest n elements we need
	average_score = sorted(average_score, key = lambda item: item[1], reverse = True)


	for i in range (0,3):
		print("topic ID: %d, average: %.4f" %(average_score[i][0], average_score[i][1]))
		print(lda.print_topic(average_score[i][0], topn=10))






Quran_list = []
OT_list = []
NT_list = []
num_Quran = num_OT = num_NT = 0

tokens_list = []

#extract the list of stop words
with open('englishST.txt', 'r') as f:
	stop_word_read = f.read()

with open('train_and_dev.tsv','r') as f:
	for line in f.readlines():
		corpus = line.split('\t')[0]
		content = line.split('\t')[1]
		content = pre_processing(content, stop_word_read)
		tokens_list+=content

		if(corpus == 'Quran'):
			num_Quran+=1
			Quran_list.append(content)
			#print(re.sub(r'[!@#$%^&*()_+{}|:"<>?,./;\'[\]\-=]+', ' ', content).lower().split())
		elif(corpus == 'OT'):
			num_OT+=1
			OT_list.append(content)
		elif(corpus == 'NT'):
			num_NT+=1
			NT_list.append(content)

N = num_Quran + num_OT + num_NT

common_texts = Quran_list+OT_list+NT_list

# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=20, id2word=common_dictionary)


# Save model to disk.
temp_file = datapath("model")
lda.save(temp_file)

# Load a potentially pretrained model from disk.
lda = LdaModel.load(temp_file)

print("corpus1:")
lda_model(lda, common_dictionary, Quran_list, num_Quran)

print("---------------------------------------------------")
print("corpus2:")
lda_model(lda, common_dictionary, OT_list, num_OT)

print("----------------------------------------------------")
print("corpus3:")
lda_model(lda, common_dictionary, NT_list, num_NT)

