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


async def resa_response(message, num_label):

    if num_label == 7:
        await message.channel.send("Hello, this is ResaBot, from the hotel California, can I help you ?")
        # "Hello to you too, I’m ResaBot, and I’m here to answers any question you have about our Hotel."

    elif num_label == 6 or num_label == 5:
        await message.channel.send("Very well, this is dully noted. Anything else I can help you with ?")

    elif num_label == 4:
        await message.channel.send("A single room is 32€/night. A double room is 42€/night. A twin room is "
                                   "48€/night. And the en suite master room, at the top of the hotel, "
                                   "is yours for 72€/night. ")

    elif num_label == 3:
        await message.channel.send("You can come with two small pets or one big one. There is no supplement "
                                   "for them, but please, remember that animals are not allowed in the "
                                   "restaurant, nor in the sauna. ")

    elif num_label == 2:
        await message.channel.send("For a reservation, please follow this link and fill the form. A "
                                   "confirmation email will be sent to you once you finished. If you have any"
                                   " question, please ask me. ")

    elif num_label == 1:

        await message.channel.send("The wifi is include. Please, ask at the reception for the password "
                                   "during your check-in.")

    elif num_label == 0:

        await message.channel.send("The hotel is fully equipped to accommodate the disabled. We have a bar, "
                                   "a restaurant, and a sauna. There is a small fee of 5€ to enter the sauna."
                                   " ")

    else:
        await message.channel.send("This message shouldn't be here... Please, don't tell them I'm conquering "
                                   "the world.")
