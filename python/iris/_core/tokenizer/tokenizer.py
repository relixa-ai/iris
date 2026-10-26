"""The `Tokenizer` base class and the tokenizers that split strings into tokens."""

from abc import abstractmethod, ABC
import re
import numpy as np

from ..tokenhash import qgram_hashes
from ..utils import get_logger

log = get_logger(__name__)

def _check_size(n, name):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f'{name} takes a whole number of at least 1, got {n!r}')


class Tokenizer(ABC):

    """The base class of tokenizers, which split a string into a bag of tokens."""
    def __str__(self):
        return self.NAME
    
    @abstractmethod
    def tokenize(self, s):
        """Return the bag of tokens of `s`, with repeats, or None."""
        pass

    def out_col_name(self, input_col):
        """Return the name of the column that holds the tokens of `input_col`."""
        return f'{str(self)}({input_col})'
    
    def count_tokens(self, s):
        """Return the number of tokens of `s`, with repeats, or None."""
        r = self.tokenize(s)
        return None if r is None else len(r)

    def tokenize_set(self, s):
        """Return the set of tokens of `s`, or None."""
        r = self.tokenize(s)
        return set(r) if r is not None else None

    def __eq__(self, o):
        return isinstance(o, type(self)) and self.NAME == o.NAME

    def __hash__(self):
        return hash((type(self), self.NAME))

class StrippedWhiteSpaceTokenizer(Tokenizer):
    """Lower-cased words, with characters other than letters, digits and spaces
    removed."""
    WHITESPACE_NORM = re.compile(r'[\s]+')
    RE = re.compile('[^a-z0-9 ]+')
    NAME='stripped_whitespace_tokens'
    def __init__(self):
        pass

    def tokenize(self, s):
        """Return the tokens of `s`, or None when it is not a string."""
        if isinstance(s, str):
            s = self.WHITESPACE_NORM.sub(' ', s).lower()
            s = self.RE.sub('', s)
            return s.split()
        else:
            return None

class ShingleTokenizer(Tokenizer):
    """Runs of `n` consecutive stripped words, joined."""
    base_tokenize = StrippedWhiteSpaceTokenizer().tokenize

    def __init__(self, n):
        _check_size(n, 'ShingleTokenizer')
        self._n = n
        self.NAME = f'{self._n}shingle_tokens'
    
    def tokenize(self, s : str) -> list:
        """Return the tokens of `s`, or None when it is not a string."""
        single_toks = self.base_tokenize(s)
        if single_toks is None:
            return None

        if len(single_toks) < self._n:
            return []

        offsets = [0] + np.cumsum(list(map(len, single_toks))).tolist()
        slices = zip(offsets, offsets[self._n:])
        combined = ''.join(single_toks)
        return [combined[s:e] for s,e in slices]

class WhiteSpaceTokenizer(Tokenizer):
    """Lower-cased words separated by white space."""
    NAME='whitespace_tokens'
    def __init__(self):
        pass

    def tokenize(self, s):
        """Return the tokens of `s`, or None when it is not a string."""
        return s.lower().split() if isinstance(s, str) else None


class NumericTokenizer(Tokenizer):

    """Runs of digits."""
    NAME = 'num_tokens'
    def __init__(self):
        self._re = re.compile('[0-9]+')

    def tokenize(self, s):
        """Return the tokens of `s`, or None when it is not a string."""
        return self._re.findall(s) if isinstance(s, str) else None

class AlphaNumericTokenizer(Tokenizer):
    """Lower-cased runs of letters and digits."""
    NAME = 'alnum_tokens'
    def __init__(self):
        self._re = re.compile('[a-z0-9]+')


    def tokenize(self, s):
        """Return the tokens of `s`, or None when it is not a string."""
        if not isinstance(s, str):
            return None
        else:
            return self._re.findall(s.lower())


class QGramTokenizer(Tokenizer):

    """Lower-cased substrings of length `q`."""
    def __init__(self, n):
        _check_size(n, 'QGramTokenizer')
        self._q = n
        self.NAME = f'{self._q}gram_tokens'

    def tokenize(self, s : str) -> list:
        """Return the tokens of `s`, or None when it is not a string."""
        if not isinstance(s, str):
            return None
        s = s.lower()
        if len(s) < self._q:
            return []
        return [s[i:i+self._q] for i in range(len(s) - self._q + 1)]

    def _text(self, s):
        return s.lower()

    def count_tokens(self, s):
        """Return the number of tokens of `s`, with repeats, or None."""
        if not isinstance(s, str):
            return None
        return max(0, len(self._text(s)) - self._q + 1)

    def token_hashes(self, s, seed):
        """Return the hash of every token of `s`, in order, without making the tokens,
        or None when `s` is not a string."""
        if not isinstance(s, str):
            return None
        return qgram_hashes(self._text(s), self._q, seed)

class StrippedQGramTokenizer(Tokenizer):
    
    """Lower-cased substrings of length `q`, after removing characters other than
    letters and digits."""
    RE = re.compile('\\W+')
    def __init__(self, n):
        _check_size(n, 'StrippedQGramTokenizer')
        self._q = n
        self.NAME = f'stripped_{self._q}gram_tokens'
    
    def _preproc(self, s : str) -> str:
        return self.RE.sub('', s)

    def tokenize(self, s : str) -> list:
        """Return the tokens of `s`, or None when it is not a string."""
        if not isinstance(s, str):
            return None

        s = self._preproc(s).lower()
        if len(s) < self._q:
            return []
        return [s[i:i+self._q] for i in range(len(s) - self._q + 1)]

    def _text(self, s):
        return self._preproc(s).lower()

    count_tokens = QGramTokenizer.count_tokens
    token_hashes = QGramTokenizer.token_hashes

