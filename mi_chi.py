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


def MI_chi(N,tokens_set,chosen_corpus,other_corpuses):

	N_1_1 = N_0_1 = N_1_0 = N_0_0 = 0


	Quran_MI = {}
	Quran_chi = {}

	for single_token in tokens_set:
		N_1_1=N_0_1=N_1_0=N_0_0=0

		for single_doc in chosen_corpus:
			if single_token in single_doc:
				N_1_1+=1
			else:
				N_0_1+=1

		for single_doc in other_corpuses:
			if single_token in single_doc:
				N_1_0+=1
			else:
				N_0_0+=1
		N_1_both = N_1_1 + N_1_0
		N_0_both = N_0_1 + N_0_0
		N_both_1 = N_1_1 + N_0_1
		N_both_0 = N_1_0 + N_0_0
		
		if(N_1_both==0 or N_both_1==0 or N_1_1==0):
			part1=0
		else:
			part1 = N_1_1/N*math.log((N*N_1_1)/(N_1_both*N_both_1),2)

		if(N_0_both==0 or N_both_1==0 or N_0_1==0):
			part2=0
		else:
			part2 = N_0_1/N*math.log((N*N_0_1)/(N_0_both*N_both_1),2)

		if(N_1_both==0 or N_both_0==0 or N_1_0==0):
			part3=0
		else:
			part3 = N_1_0/N*math.log((N*N_1_0)/(N_1_both*N_both_0),2)

		if(N_0_both==0 or N_both_0==0 or N_0_0==0):
			part4=0
		else:
			part4 = N_0_0/N*math.log((N*N_0_0)/(N_0_both*N_both_0),2)

		mutual_information = part1 + part2 + part3 + part4

		Quran_MI[single_token] = mutual_information

		if (N_both_1==0 or N_1_both==0 or N_both_0==0 or N_0_both==0):
			chi_squared=0
		else:
			chi_squared = (N_1_1 + N_1_0 + N_0_1 + N_0_0)*math.pow(N_1_1*N_0_0-N_1_0*N_0_1,2)/ \
				(N_both_1*N_1_both*N_both_0*N_0_both)

		Quran_chi[single_token] = chi_squared


	Quran_MI_list = sorted(Quran_MI.items(), key = lambda item: item[1], reverse = True)
	Quran_chi_list = sorted(Quran_chi.items(), key = lambda item: item[1], reverse = True)

	for i in range(0,10):
		print("%s,%.3f" %(Quran_MI_list[i][0], Quran_MI_list[i][1]))


	for i in range(0,10):
		print("%s,%.3f" %(Quran_chi_list[i][0], Quran_chi_list[i][1]))




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

tokens_set = set(tokens_list) #each token only once

#Quran corpus
print("Quran")
MI_chi(N,tokens_set,Quran_list,OT_list+NT_list)

print("----------------------------")
print("OT")
MI_chi(N,tokens_set,OT_list,Quran_list+NT_list)

print("----------------------------")
print("NT")
MI_chi(N,tokens_set,NT_list,Quran_list+OT_list)
