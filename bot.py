import discord
import torch
import transformers

from config import TOKEN, BATCH_SIZE, TEXT_MAX_LENGTH
from src.functions import text_preprocessing

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

            if num_label == 7:
                await message.channel.send("Hello, this is ResaBot, from the hotel California, can I help you ?")
                # "Hello to you too, I’m ResaBot, and I’m here to answers any question you have about our Hotel."

            elif num_label == 6 or preds[0].item() == 5:
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


# ____INIT APP

client = MyClient()
client.run(TOKEN)
