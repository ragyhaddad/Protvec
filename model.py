#!/bin/bash/env python
import sys, os,json 
from gensim.models import word2vec
import numpy as np 
from Bio import SeqIO 
from gensim.models.keyedvectors import KeyedVectors # Convert Binary Model to Txt

# Test String
ts = 'MLSGHJKLTTSYTSLKM'  

# Split Data to ngrams
# for example if you have 'ATSKLGH' --> [['ATS','KLG'],['TSK','LGH],['SKL']] --> no overlapping seqs
def split_ngrams(seq, n):
    # Fancy way for splitting in kmers
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

# Generate corpusfile -- Take a fasta file and use it as a training set
# In our case we will use either the swissprot fasta database or pfam databases
def generate_corpusfile(corpus_fname,n,out):
    f = open(out,"w") 
    for r in SeqIO.parse(corpus_fname,"fasta"):
        ngram_patterns = split_ngrams(r.seq,n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n") # Here we are taking in all the sequences and splitting them into kmers
            sys.stdout.write(".") # this is a loading thing to know its working
            # The output will be each sequence split into kmers 
    f.close() 

# Class For setting up word2vec 
# Take the word2vec class and set it up to make protvec
# This class is based on the class used in the biovec package on github with slight mods
class ProtVec(word2vec.Word2Vec):
    def __init__(self,corpus_fname=None,corpus=None,n=3,size=100,out="corpus.txt",sg=1, window=25, min_count=1, workers=48):
        skip_gram = 1 # Skip gram true
        self.n = n 
        self.size = size 
        self.corpus_fname = corpus_fname 
        self.sg = (skip_gram) 
        self.window = window
        self.workers = workers 
        self.out = out 
        self.vocab = min_count
        if corpus is not None and corpus_fname is None:
            raise Exception("Corpus or File Name is Incorrect") 
        if corpus_fname is not None:
            if not os.path.isfile(out):
                print("-- Generating Corpus")
                generate_corpusfile(corpus_fname,n,out) 
            else: 
                print("-- Corpus File Found...") 
        # Set Up Corpus 
        self.corpus = word2vec.Text8Corpus(out) 
        print("-- Corpus Setup Successful... ")
    # This is Copied From Docs - This just initializes the class then trains/saves - You can do it manually  
    def word2vec_init(self, ngram_model_fname):
        print('-- Initializing Word2Vec Class..')
        print('-- Training Model...')
        model = word2vec.Word2Vec.__init__(self, self.corpus, size=self.size, sg=self.sg, window=self.window, min_count=self.vocab, workers=self.workers)
        model.wv.save_word2vec_format(ngram_model_fname)
        model.save('test-class.mdl') # This will be removed later
        print('-- Saving Model Weights to: %s ' % (ngram_model_fname))

    def to_vecs(self,seq,ngram_vectors):
        ngrams_seq = split_ngrams(seq,self.n) 
        protvec = np.zeros(self.size,dtype=np.float32)
        for index in xrange(len(seq) + 1 - self.n):
            ngram = seq[index:index + self.n]
            if ngram in ngram_vectors:
                ngram_vector = ngram_vectors[ngram]
                protvec += ngram_vector
        return normalize(protvec) # Normalize Like they did in the paper 

    # Take a fasta file and return the n_gram vectors
    # return: String <Array>
    def get_ngram_vectors(self, file_path):
        ngram_vectors = {}
        vector_length = None
        with open(file_path) as infile:
            for line in infile:
                line = line.strip()
                line_parts = line.rstrip().split()   
                # skip first line with metadata in word2vec text file format
                if len(line_parts) > 2:     
                    ngram, vector_values = line_parts[0], line_parts[1:]          
                    ngram_vectors[ngram] = np.array(map(float, vector_values), dtype=np.float32)
        return ngram_vectors

    # Load The trained Model Weights
    def load_protvec(self,model_fname):
        return word2vec.Word2Vec.load(model_fname)

    # Normalize the Weights of the embedding as done in the paper
    def normalize(x):
        return x / np.sqrt(np.dot(x, x))

# Driver
def main():
    # Run Model and Save it
    # model = ProtVec(corpus_fname=sys.argv[1])
    # model.word2vec_init('test.mdl') 
    # model.save('test.mdl')
    
    # To Load Weights into text 
    from gensim.models import Word2Vec
    model = ProtVec()
    model = model.load("3-gram-final.mdl.bin")
    # model.load("3-gram-final.mdl.bin")
    #model.get_ngram_vectors(ts)
    # model.wv.save_word2vec_format('test-3core.txt', binary=False)
if __name__ == '__main__':
    main()





