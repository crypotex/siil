from nltk import word_tokenize
from collections import Counter

import re
from nltk.tokenize import word_tokenize
from string import punctuation


def preprocess(text):
    text = re.sub(r"([%s])" % punctuation, r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def bigrams(text_tokens):
    n2grams = list(zip(text_tokens[:-1], text_tokens[1:]))
    n2grams = ["__".join(i) for i in n2grams]
    return " ".join(n2grams)


def prep_text(texts, lower=False):
    unis = []
    bis = []
    for line in texts:
        if lower:
            text = preprocess(line.lower())
        else:
            text = preprocess(line)
        text_bigrams = bigrams(word_tokenize(text))
        unis.append(text)
        bis.append(text_bigrams)
    return unis, bis


def concatenate_texts(texts, lower=False):
    unis, bis = prep_text(texts)
    outtexts = list(" ".join(i) for i in zip(unis, bis))
    return outtexts


def write_texts(texts, outf):
    with open(outf, "w") as f:
        for line in texts:
            f.write("%s\n" % line)
    print("done")