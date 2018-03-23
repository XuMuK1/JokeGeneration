import os
import torch
import nltk
import csv
import itertools
import operator
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

beginToken="\n###beginjoke###\n"
endToken="\n###endjoke###\n"
unknownToken="<??>"

class Corpus(object):

    

    def __init__(self, path, vocabularySize):
        self.dictionary = self.trainDictionary(os.path.join(path, 'all.txt'),vocabularySize)

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

        

    def trainDictionary(self, path, vocabulary_size=8000,min_sent_characters=0):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print("Learning dictionary...")
        print("Reading CSV file...",path)
        with open(path, 'rt') as f:
        	# Split jokes into sentences
        	# Filter sentences
        	#sentences = [s for s in f if len(s) >= min_sent_characters]
        	#sentences = [s for s in sentences if "http" not in s]
        	# Append SENTENCE_START and SENTENCE_END
        	#print(nltk.word_tokenize(sentences[0]))
        	sentences = [x.split() for x in f if len(x)>1]
        print("Parsed %d sentences." % (len(sentences)))

    	# Tokenize the sentences into words
        tokenized_sentences = sentences#[nltk.word_tokenize(sent[1:-1]) for sent in sentences]

    	# Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        tokenized_sentences = [[beginToken]+ x + [endToken] for x in tokenized_sentences]
    	# Get the most common words and build index_to_word and word_to_index vectors
        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
        dictionary= Dictionary()
        dictionary.idx2word = ["<MASK/>", unknownToken,beginToken,endToken] + [x[0] for x in sorted_vocab]
        dictionary.word2idx = dict([(w, i) for i, w in enumerate(dictionary.idx2word)])
        
        return dictionary
    
    def tokenize(self,path,min_sent_characters=0):

        assert os.path.exists(path)
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        print("Reading file...",path)
        with open(path, 'rt') as f:
        	# Split jokes into sentences
        	#sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in f if len(x)>0])
        	# Filter sentences
        	#sentences = [s for s in sentences if len(s) >= min_sent_characters]
        	#sentences = [s for s in sentences if "http" not in s]
        	# Append SENTENCE_START and SENTENCE_END
        	sentences = [x.split() for x in f if len(x)>1]
        print("Parsed %d sentences." % (len(sentences)))
        
        # sequenize file content
        tokenized_sentences = [[beginToken]+ x + [endToken] for x in sentences]
        tokens=np.sum([len(sent) for sent in tokenized_sentences])
        ids = torch.LongTensor(np.zeros(tokens))
        token = 0
        for sent in tokenized_sentences:
            for word in sent:
                try:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                except:
                    ids[token]=1    
                token += 1

        return ids