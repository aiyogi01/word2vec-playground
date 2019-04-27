import re
from typing import List


PATTERN_HTML = re.compile(r"<[^>]+>")
PATTERN_PUNCTUATION = re.compile(r"[-.,:;!?\"#$%&()*+/<=>@\[\]\\^_`{|}~\t\n]")
PATTERN_WHITESPACE = re.compile(r"\s+")


def to_lower(text: str) -> str:
    """
    Convert text to lower case.
    """
    return text.lower()


def remove_html(text: str) -> str:
    """
    Remove HTML tags.
    """
    return re.sub(PATTERN_HTML, "", text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation marks.
    """
    return re.sub(PATTERN_PUNCTUATION, " ", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize white space.
    """
    return re.sub(PATTERN_WHITESPACE, " ", text).strip()


def preprocess(text: str) -> str:
    """
    Preprocess text by applying a pipeline.
    """
    pipeline = [
        remove_html,
        remove_punctuation,
        to_lower,
        normalize_whitespace
    ]
    for func in pipeline:
        text = func(text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Preprocess and tokenize text.
    """
    text = preprocess(text)
    return text.split(" ")
