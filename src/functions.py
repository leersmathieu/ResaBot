import contractions
import unidecode
from bs4 import BeautifulSoup

from config import TOKEN, BATCH_SIZE, TEXT_MAX_LENGTH


# ____PRE PROCESSING

def strip_html_tags(text: str):
    """Remove html tags from text."""

    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")

    return stripped_text


def expand_contractions(text: str):
    """Expand shortened words, e.g. 'don't' to 'do not'."""

    text = contractions.fix(text)
    return text


def remove_accented_chars(text):
    """Remove accented characters from text, e.g. café."""

    text = unidecode.unidecode(text)
    return text


def remove_whitespace(text: str):
    """Remove extra whitespaces from text."""

    text = text.strip()
    return " ".join(text.split())


def limit_n_words(text: str, limit: int = TEXT_MAX_LENGTH):
    """Limit a text to 256 words."""

    text = text.split()[:limit]
    return " ".join(text)


def text_preprocessing(text: str):
    # Preprocess
    text = strip_html_tags(text)
    text = remove_whitespace(text)
    text = remove_accented_chars(text)
    text = expand_contractions(text)
    text = limit_n_words(text)
    text = text.lower()

    return text
