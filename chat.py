import telebot
import codecs
import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with codecs.open('intents.json', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Задайте вопрос. Если я не смогу на него ответить, то просьба обратиться по этой ссылке smois@77gmail.com :)")
while True:
    sentence = input("Вы:  ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Ваш вопрос вышел за рамки моей компетенции. Пожалуйсте, перефразируйте его.")
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

def start(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi')


def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def run_bot(update: Update, _: CallbackContext) -> None:
    response = main(update.message.text)
    update.message.reply_text(sentence)
    print(update.message.text)
    print(response)
    print()




def main():
    """Start the bot."""
    updater = Updater("1577959536:AAF3gn-oFfpxnAr1E8pnhy6yU0thyiaBsmI")
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, run_bot))

    updater.start_polling()
    updater.idle()

    main()













