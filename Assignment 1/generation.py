import numpy as np
import re
from sklearn.linear_model import LinearRegression
import argparse
from language_model import NgramModel , EstimateParams
from tokeniser import tokeniser

def GoodTuringGeneration(corpus_path , sentence ,words, Ng = 3 ):
    tokenised_text = tokeniser(sentence)
    trigram = NgramModel(3 , corpus_path)

    max_freq = max(trigram.values())
    freqoffreq = np.zeros((max_freq+1) , dtype=np.int32)

    # find the frequency of each frequency
    for key in trigram:
        freqoffreq[trigram[key]] += 1
    
        # Normalising the freqoffreq
    for r in range(2 , max_freq+1):
        # find nearest nonzero indice on left
        t = r-1
        while t > 0 and freqoffreq[t] == 0:
            t -= 1
        q = r+1
        while q < max_freq and freqoffreq[q] == 0:
            q += 1
        
        freqoffreq[r] = freqoffreq[r] / (q-t)
    
    # handle Nr = 0
    rs = np.arange(1 , max_freq+1) # r > 0
    N = 0
    for r in rs:
        # print(freqoffreq[r])
        if freqoffreq[r] == 0:
            freqoffreq[r] = freqoffreq[r-1]
        
        N += r*freqoffreq[r]
    
    # Now , for Linear Good Turing Estimate (LGT) , estimate b
    logNr = np.log(np.array(freqoffreq[1:]))
    logr = np.log(rs)
    model  = LinearRegression()
    model.fit(logr.reshape(-1,1) , logNr)
    b = model.coef_[0]
    a = model.intercept_

    # Now calculate prob of each r
    recounted_rs = np.zeros((max_freq+1))
    recounted_rs[0] = np.exp(a) 
    for r in rs:
        recounted_r = (r+1)*((1 + 1/r)**b)
        recounted_rs[r] = recounted_r

    tri = tuple(tokenised_text[0][len(tokenised_text[0]) - 3 : len(tokenised_text[0])])
   
    if tri in trigram:
        r = trigram[tri]
    else:
        r = 0
    num = recounted_rs[r]
    # Now for all possible trigrams , calculate sum of recounted rs
    den = 0
    for word in words:
        bi = tuple(tokenised_text[0][len(tokenised_text[0]) - 3 : len(tokenised_text[0])-1] + [word])
        if bi in trigram:
            den += recounted_rs[trigram[bi]]
        else:
            den += recounted_rs[0]
    final_prob = num/den

    return final_prob

def LinearInterpolationGeneration(corpus_path , sentence , Ng = 3):
    tokenised_text = tokeniser(sentence)
    params = EstimateParams(corpus_path)

    trigram = NgramModel(3 , corpus_path)
    bigram = NgramModel(2 , corpus_path)
    unigram = NgramModel(1 , corpus_path)

    keys = trigram.keys()
    keys = list(keys)
    corpus_size = len(keys)

    tri = tuple(tokenised_text[0][len(tokenised_text[0]) - 3 : len(tokenised_text[0])])
    bi = tuple(tokenised_text[0][len(tokenised_text[0]) - 2 : len(tokenised_text[0])])
    uni = tuple(tokenised_text[0][len(tokenised_text[0]) - 1 : len(tokenised_text[0])])

    prob_tri = params[2]*trigram[tri]/bigram[bi] if (bi in bigram and tri in trigram) else 0
    prob_bi = params[1]*bigram[bi]/unigram[uni] if (uni in unigram and bi in bigram) else 0
    prob_uni = params[0]*unigram[uni]/corpus_size if (uni in unigram) else 0
    final_prob = prob_tri + prob_bi + prob_uni

    return final_prob

def NgramGeneration(corpus_path , sentence , Ng = 3):
    ngram = NgramModel(Ng , corpus_path)
    if Ng > 1:
        lessgram = NgramModel(Ng , corpus_path , Ng-1)

    tokenised_text = tokeniser(sentence)
    
    # print(ngram)
    prob = 0
    n_gram_tuple = tuple(tokenised_text[0][len(tokenised_text[0]) - Ng : len(tokenised_text[0])])
    if Ng > 1:
        lessgram_tuple = tuple(tokenised_text[0][len(tokenised_text[0]) - Ng  : len(tokenised_text[0])-1])
        if (n_gram_tuple in ngram) and (lessgram_tuple in lessgram) :
            prob = (ngram[n_gram_tuple] / lessgram[lessgram_tuple])
    else:
        corpus_size = sum(ngram.values())
        if n_gram_tuple in ngram:
            prob = ngram[n_gram_tuple] / corpus_size
    
    return prob


def WordPrediction(sentence , corpus_path , ng , k , model = None):
    # find all words in the corpus_path
    text = open(corpus_path, 'r').read()
    tokenised_corpus = tokeniser(text)
    words = []
    punctuations = ['!', '?','.' , ',' , ';', '(', ')', '[', ']', '{', '}', '"', "'"]
    for sen in tokenised_corpus:
        # remove punctuations
        for punctuation in punctuations:
            while punctuation in sen:
                sen.remove(punctuation)
        words += sen
    
    words = list(set(words))
    print("Number of words in the corpus: " , len(words))
    # find the 'k' most probable words , use max-heap with size 'k'
    probs = []
    for word in words:
        prob = 0
        if model == 'g':
            prob = GoodTuringGeneration(corpus_path , sentence + ' ' + word , words , ng)
        elif model == 'i':
            prob = LinearInterpolationGeneration(corpus_path , sentence + ' ' + word , ng)
        else:
            prob = NgramGeneration(corpus_path , sentence + ' ' + word , ng)
        
        probs.append(prob)
    
    probs = np.array(probs)
    max_indices = np.argsort(probs)[-k:]
    max_words = []
    for index in reversed(max_indices):
        max_words.append({"word" : words[index] , "prob" : probs[index]})
    
    return max_words

def main():
    # make arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='sample.txt', help='path to corpus')
    parser.add_argument('--model' , type=str , default='i' , help='Model to use')
    parser.add_argument('--k' , type=int , default=2 , help='Number of words to predict')
    parser.add_argument('--ng' , type=int , default=3 , help='N-gram model to use')

    args = parser.parse_args()

    # take input from user
    sentence = input("Enter sentence: ")
    max_words = WordPrediction(sentence , args.corpus_path , args.ng , args.k , args.model)
    for word in max_words:
        # print(f'${word['word']} ${word['prob']}')
        print(word['word'] , word['prob'])
    

if __name__ == '__main__':
    main()