import re

class Tokenizer:
    emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

    regex_str = [
        emoticons_str,
        r'<[^>]+>',
        r'(?:@[\w_]+)',
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',

        r'(?:(?:\d+,?)+(?:\.?\d+)?)',
        r"(?:[a-z][a-z'\-_]+[a-z])",
        r'(?:[\w_]+)',
        r'(?:\S)'
    ]

    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

    @classmethod
    def tokenize(s):
        return tokens_re.findall(s)

    def preprocess(s, lowercase=False):
        tokens = tokenize(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        return tokens
