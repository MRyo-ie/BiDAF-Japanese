import nltk

nltk.download('punkt')

def tokenize(sequence, do_lowercase):
    """
    英文を単語分する。
    Tokenizes the input sequence using nltk's word_tokenize function, replaces two single quotes with a double quote
    """
    if do_lowercase:
        tokens = [token.replace("``", '"').replace("''", '"').lower()
                  for token in nltk.word_tokenize(sequence)]
    else:
        tokens = [token.replace("``", '"').replace("''", '"')
                  for token in nltk.word_tokenize(sequence)]
    return tokens
