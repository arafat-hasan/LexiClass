from __future__ import annotations

import importlib
import re
from typing import List

try:
    _ICU_AVAILABLE = importlib.util.find_spec('icu') is not None  # type: ignore[attr-defined]
except Exception:
    _ICU_AVAILABLE = False


class RegularExpressions:
    """Regular expression patterns for tokenization."""

    MAIL_TO = r'\bmailto\:[^@\s]+@[^@\s]+\.[^@\s]\b'
    URI = r'\b[^\W\d_]+\:\/\/[^\s<>"]+\b'
    WORD = r'\b[^\W][\w.-@\/\\]+[^\W]\b'
    NUMBER = r'\b\-?\+?\d+\.?\d+?\b'


class ICUTokenizer:
    """Locale-aware tokenizer using ICU word boundaries with regex fallback.

    Serialization is lightweight; ICU internals are re-initialized on load.
    """

    def __init__(self, locale: str = 'en') -> None:
        self.locale = locale
        self.break_iterator = None
        self._fallback_token_pattern = None
        self._fallback_filter_pattern = None

        if _ICU_AVAILABLE:
            try:
                icu_mod = importlib.import_module('icu')
                bi_cls = getattr(icu_mod, 'BreakIterator', None)
                loc_ctor = getattr(icu_mod, 'Locale', None)
                if bi_cls is not None and loc_ctor is not None:
                    self.break_iterator = bi_cls.createWordInstance(loc_ctor(locale))  # type: ignore
            except Exception:
                self.break_iterator = None

        if self.break_iterator is None:
            self._fallback_token_pattern = re.compile(r'(?u)' + RegularExpressions.WORD)
            self._fallback_filter_pattern = re.compile(r'(?u)' + '|'.join([
                RegularExpressions.NUMBER,
                RegularExpressions.MAIL_TO,
                RegularExpressions.URI,
            ]))

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if self.break_iterator is not None:
            self.break_iterator.setText(text)
            tokens: List[str] = []
            tstart = 0
            for tend in self.break_iterator:
                token = text[tstart:tend].strip()
                if token:
                    tokens.append(token.lower())
                tstart = tend
            return tokens
        text = text.lower()
        tokens = self._fallback_token_pattern.findall(text) if self._fallback_token_pattern else []
        return [t for t in tokens if not (self._fallback_filter_pattern and self._fallback_filter_pattern.match(t))]

    def __getstate__(self):
        return {'locale': self.locale}

    def __setstate__(self, state):
        self.locale = state.get('locale', 'en')
        self.break_iterator = None
        self._fallback_token_pattern = None
        self._fallback_filter_pattern = None
        if _ICU_AVAILABLE:
            try:
                icu_mod = importlib.import_module('icu')
                bi_cls = getattr(icu_mod, 'BreakIterator', None)
                loc_ctor = getattr(icu_mod, 'Locale', None)
                if bi_cls is not None and loc_ctor is not None:
                    self.break_iterator = bi_cls.createWordInstance(loc_ctor(self.locale))  # type: ignore
            except Exception:
                self.break_iterator = None
        if self.break_iterator is None:
            self._fallback_token_pattern = re.compile(r'(?u)' + RegularExpressions.WORD)
            self._fallback_filter_pattern = re.compile(r'(?u)' + '|'.join([
                RegularExpressions.NUMBER,
                RegularExpressions.MAIL_TO,
                RegularExpressions.URI,
            ]))


