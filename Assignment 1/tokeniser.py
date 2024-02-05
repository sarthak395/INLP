# %%
import numpy as np
import re


# %%
def sentencetokeniser(text):
    # split the text into sentences using regex a new line is encountered
    sentences = re.split(r'\n\n', text)
    # remove empty strings
    sentences = list(filter(None, sentences))
    # clean out the \n
    sentences = [sentence.replace('\n', ' ') for sentence in sentences]
    return sentences

# %%
def wordtokeniser(sentence):
    # split the sentences into words using regex a space is encountered
    words = re.split(r' ', sentence)
    # remove empty strings
    words = list(filter(None, words))
    return words

# %%
def replacenumbers(words):
    # replace numbers with a special token
    for i in range(len(words)):
        # starts and ends with a number with only numbers in between
        if re.match(r'^\d+$', words[i]):
            words[i] = '<NUM>'
    return words

# %%
def detectpunctuations(text):
    punctuations = ['!', '?',]
    for punctuation in punctuations:
        text = text.replace(punctuation, ' ' + punctuation + '\n\n')
    
    seperators = [',' , ';', '(', ')', '[', ']', '{', '}', '"', "'"]
    for seperator in seperators:
        text = text.replace(seperator, ' ' + seperator + ' ')
    
    # if before '.' , there is a lowercase letter and after tricky1 there is a space , then it a fullstop . Only then replace it with a fullstop and a new line
    text = re.sub(r'([a-z])\. ', r'\1 .\n\n', text)
    text = re.sub(r'([a-z]) \. ', r'\1 .\n\n', text)
    text = re.sub(r'([a-z])\.\n', r'\1 .\n\n', text)
    text = re.sub(r'([a-z])\.$', r'\1 .\n\n', text)
    return text

# %%
def replacemailids(words):
    # replace email ids with a special token
    for i in range(len(words)):
        if re.match(r'^\w+@\w+\.\w+$', words[i]):
            words[i] = '<MAILID>'
    return words

# %%
def replaceURLs(words):
    # replace URLs with a special token
    starts = ['http://', 'https://', 'www.']
    for i in range(len(words)):
        for start in starts:
            if words[i].startswith(start):
                words[i] = '<URL>'
    return words

# %%
def replacehash(words):
    # replace hashtags with a special token
    for i in range(len(words)):
        # if word starts with # and contains atleast 1 letter
        if words[i].startswith('#') and not re.match(r'^\d+$', words[i][1:]):
            words[i] = '<HASHTAG>'
    return words

# %%
def replacementions(words):
    # replace mentions with a special token
    for i in range(len(words)):
        # if word starts with @ and contains atleast 1 letter
        if words[i].startswith('@') and not re.match(r'^\d+$', words[i][1:]):
            words[i] = '<MENTION>'
    return words

# %%
def replaceids(words):
    # replace ids with a special token
    for i in range(len(words)):
        # if word starts with # and contains atleast 1 letter
        if words[i].startswith('#') and re.match(r'^\d+$', words[i][1:]):
            words[i] = '<ID>'
    return words

def tokeniser(text):
    newtext = detectpunctuations(text)
    sns = sentencetokeniser(newtext)
    # FOR ALL SENTENCES , SPLIT INTO WORDS
    allsns = []
    for sn in sns:
        words = wordtokeniser(sn)

        # REPLACE WITH A SPECIAL TOKEN
        words = replacenumbers(words)
        words = replacemailids(words)
        words = replaceURLs(words)
        words = replacehash(words)
        words = replacementions(words)
        words = replaceids(words)

        allsns.append(words)

    return allsns

def main():
    text = input("Enter text: ")
    tokenisedtext = tokeniser(text)
    print(tokenisedtext)

if __name__ == "__main__":
    main()



