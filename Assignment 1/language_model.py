import numpy as np
import re
from sklearn.linear_model import LinearRegression
from tokeniser import tokeniser
import argparse

def NgramModel(N , corpus_path , r = None):
    text = open(corpus_path, 'r').read()
    tokenised = tokeniser(text)

    # N-gram model
    tags = ['<BOS>' , '<EOS>']

    # create a dictionary of all the N-grams
    ngrams = {}
    for sentence in tokenised:
        # add N-1 <BOS> tags in the beginning
        for i in range(N-1):
            sentence.insert(0, tags[0])
        # add 1 <EOS> tag in the end
        sentence.append(tags[1])

        if r is None:
            for i in range(len(sentence)-N+1):
                ngram = tuple(sentence[i:i+N])
                if ngram not in ngrams:
                    ngrams[ngram] = 0
                ngrams[ngram] += 1
        else:
            for i in range(len(sentence)-r+1):
                ngram = tuple(sentence[i:i+r])
                if ngram not in ngrams:
                    ngrams[ngram] = 0
                ngrams[ngram] += 1
    
    return ngrams

def NgramProb(sentence , n , corpus_path):
    ngram = NgramModel(n , corpus_path)
    if n > 1:
        lessgram = NgramModel(n , corpus_path , n-1)

    tokenised_text = tokeniser(sentence)
    
    num_sentences = len(tokenised_text)

    for i in range(num_sentences):
        # add N-1 <BOS> tags in the beginning
        for j in range(n-1):
            tokenised_text[i].insert(0, '<BOS>')
        # add 1 <EOS> tag in the end
        tokenised_text[i].append('<EOS>')
    # print(ngram)
    prob = 1
    for sentence in tokenised_text:
        for i in range(len(sentence)-n+1):
            ngram_tuple = tuple(sentence[i:i+n])
            lessrgam_tuple = tuple(sentence[i:i+n-1])

            if (ngram_tuple in ngram) and (lessrgam_tuple in lessgram) :
                prob *= (ngram[ngram_tuple] / lessgram[lessrgam_tuple])
            else:
                prob *= 1e-6
    
    return prob

def EstimateParams(corpus_path):
    params = [0 for i in range(3)]
    trigram = NgramModel(3 , corpus_path)
    bigram = NgramModel(2 , corpus_path)
    unigram = NgramModel(1 , corpus_path)

    keys = trigram.keys()
    keys = list(keys)
    corpus_size = len(keys)

    # for all possible trigrams 
    for key in keys:
        values = [0.0 for i in range(3)]
        # if value doesn't exist , then it is 0

        if (key[0] , key[1]) in bigram and bigram[(key[0] , key[1])] > 1:
            values[2] = (trigram[key] - 1)/ (bigram[(key[0] , key[1])]-1)

        if (key[1],) in unigram and unigram[(key[1],)] > 1:
            values[1] = (bigram[(key[1] , key[2])] - 1)/ (unigram[(key[1],)]-1)
        
        if (key[2],) in unigram:
            values[0] = (unigram[(key[2],)] - 1)/ (corpus_size-1)

    # find the maximum value index
        max_index = np.argmax(values)
        params[max_index] += trigram[key]
    
    # normalise the params
    params = np.array(params)
    params = params/np.sum(params)
    return params

def LinearInterpolation(corpus_path , sentence , N = 3 ):
    tags = ['<BOS>' , '<EOS>']
    tokenised_text = tokeniser(sentence)
    
    # tokenised_text is a 2D list with each row representing a sentence and each column representing a word
    num_sentences = len(tokenised_text)
    for i in range(num_sentences):
        # add N-1 <BOS> tags in the beginning
        for j in range(N-1):
            tokenised_text[i].insert(0, tags[0])
        # add 1 <EOS> tag in the end
        tokenised_text[i].append(tags[1])

    params = EstimateParams(corpus_path)
    trigram = NgramModel(3 , corpus_path)
    bigram = NgramModel(2 , corpus_path)
    unigram = NgramModel(1 , corpus_path)

    keys = trigram.keys()
    keys = list(keys)
    corpus_size = sum(unigram.values())

    # break the sentence into trigrams
    final_prob = 1
    for j in range(num_sentences):
        for i in range(len(tokenised_text[j])-N+1):
            tri = tuple(tokenised_text[j][i:i+N])
            bi = tuple(tokenised_text[j][i+1:i+N])
            uni = tuple(tokenised_text[j][i+2:i+N])

            prob_tri = params[2]*trigram[tri]/bigram[bi] if (bi in bigram and tri in trigram) else 0
            prob_bi = params[1]*bigram[bi]/unigram[uni] if (uni in unigram and bi in bigram) else 0
            prob_uni = params[0]*unigram[uni]/corpus_size if (uni in unigram) else 0
            prob = prob_tri + prob_bi + prob_uni
            final_prob *= prob
    
    return final_prob

def GoodTuringMethod(corpus_path, sentence , Ng = 3):
    tags = ['<BOS>' , '<EOS>']
    tokenised_text = tokeniser(sentence)
    text = open(corpus_path, 'r').read()
    tokenised_corpus = tokeniser(text)
    words = []

    for sen in tokenised_corpus:
        words += sen
    
    words = list(set(words))
    
    # tokenised_text is a 2D list with each row representing a sentence and each column representing a word
    num_sentences = len(tokenised_text)
    for i in range(num_sentences):
        # add N-1 <BOS> tags in the beginning
        for j in range(Ng-1):
            tokenised_text[i].insert(0, tags[0])
        # add 1 <EOS> tag in the end
        tokenised_text[i].append(tags[1])
    
    trigram = NgramModel(3 , corpus_path)
    # max value of trigram.values()
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
    # prob = np.zeros((max_freq+1))
    # prob[0] = np.exp(a) / N
    recounted_rs = np.zeros((max_freq+1))
    recounted_rs[0] = np.exp(a) 
    for r in rs:
        recounted_r = (r+1)*((1 + 1/r)**b)
        recounted_rs[r] = recounted_r


    # Now , we can calculate prob of sentence
    final_prob = 1
    for j in range(num_sentences):
        for i in range(len(tokenised_text[j])-Ng+1):
            tri = tuple(tokenised_text[j][i:i+Ng])
            if tri in trigram:
                r = trigram[tri]
            else:
                r = 0
            num = recounted_rs[r]
            # Now for all possible trigrams , calculate sum of recounted rs
            den = 0
            for word in words:
                bi = tuple(tokenised_text[j][i:i+Ng-1] + [word])
                if bi in trigram:
                    den += recounted_rs[trigram[bi]]
                else:
                    den += recounted_rs[0]
            prob = num/den
            final_prob *= prob
    
    return final_prob

def main():
    # make arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='Corpus/Pride and Prejudice - Jane Austen.txt', help='path to corpus')
    parser.add_argument('--model' , type=str , default='i' , help='Model to use')

    args = parser.parse_args()

    # take input from user
    sentence = input("Enter sentence: ")

    # calculate probability
    prob = 0
    if args.model == 'i':
        prob = LinearInterpolation(args.corpus_path , sentence)
    elif args.model == 'g':
        prob = GoodTuringMethod(args.corpus_path , sentence)
    
    print("Score: " , prob)

if __name__ == '__main__':
    main()
