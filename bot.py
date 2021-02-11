import discord
import torch
import contractions
import unidecode
from bs4 import BeautifulSoup
import transformers

print(torch.cuda.is_available())

model = torch.load("model/resa_model_100")
print(model.eval())

# ____CONSTANT

TOKEN = "PUT YOUR DISCORD TOKEN HERE"
TEXT_MAX_LENGTH = 256
BATCH_SIZE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = {
    0: 'longtalk_accomodations',
    1: 'longtalk_internet',
    2: 'longtalk_make_reservation',
    3: 'longtalk_pets',
    4: 'longtalk_price',
    5: 'smalltalk_confirmation_no',
    6: 'smalltalk_confirmation_yes',
    7: 'smalltalk_greetings_hello'
}


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


# ____BERT TOKENIZER

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


# ____CHATTERBOT

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, message):

        if message.author == self.user:
            return

        elif message.content == 'ping':
            await message.channel.send('pong')

        else:
            # don't respond to ourselves
            message_preprocessed = text_preprocessing(message.content)
            print(message_preprocessed)

            sample_inputs = tokenizer(message_preprocessed, return_tensors="pt").to(device)

            outputs = model(
                input_ids=sample_inputs.input_ids,
                token_type_ids=None,
                attention_mask=sample_inputs.attention_mask
            )

            _, preds = torch.max(outputs[0], dim=1)
            num_label = preds[0].item()

            # print(labels[num_label])
            await message.channel.send(labels[num_label])


# ____INIT APP

client = MyClient()
client.run(TOKEN)
