import discord
import torch
import transformers

from config import TOKEN, BATCH_SIZE, TEXT_MAX_LENGTH
from src.functions import text_preprocessing
from src.functions import resa_response

print(torch.cuda.is_available())

model = torch.load("model/resa_model_100")
print(model.eval())

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
            print(num_label, labels[num_label])

            await resa_response(message, num_label)


# ____INIT APP

client = MyClient()
client.run(TOKEN)
