from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

import numpy as np
import random
import sys

class Globals():

    """

    Store global values for use in data processing and the model

    """

    maxlen = 40
    step = 3
    filename = "lyric.txt"
    range_num = 60
    epochs = 1

class Data:

    """

    A dictionary of all processed data values

    text<string>
        The text corpus that is currently being processed

    chars set<char>
        A collection of all chars in text

    char_indices D<char:integer>
        A mapping from every char to an integer

    indices_char D<integer:char>
        A mapping from every integer to a char

    sentences List<String>
        segments text data into pieces of maxlen

    next_chars
        every char every distance of i +  max_len

    X
        pass

    y
        pass


    """

    def __init__(self, **kwargs):
        for key, value  in kwargs.items():
            self.__dict__[key] = value

    @staticmethod
    def generate(filename):

        """

        Generates all data items needed to train the model

        """

        def read_text(f):

            """

            Reads text from file

            f<string> -> string

            """
            with open(f, 'r') as infile:
                return infile.read().lower()

        text = read_text(filename)

        def create_chars_dicts():

            """

            Finds all unique chars in text and then creates dictionaries mapping those to text

            -> set<char>, dic<char:int>,
            """

            chars = sorted(list(set(text)))

            char_indices = {c:i for i, c in enumerate(chars)}
            indices_char = {i:c for i, c in enumerate(chars)}

            return chars, char_indices, indices_char

        chars, char_indices, indices_char = create_chars_dicts()

        def generate_sentences_and_next_chars():


            """

            Segments text so that it can be trained

            -> List<String>, List <Char>

            """

            sentences = []
            next_chars = []

            for i in range(0,len(text) - Globals.maxlen, Globals.step):
                sentences.append(text[i: i + Globals.maxlen])
                next_chars.append(text[i + Globals.maxlen])

            return sentences, next_chars

        sentences, next_chars = generate_sentences_and_next_chars()

        def vectorize():

            """
            Creates tensor which represents data

            -> Tensor

            """

            X = np.zeros((len(sentences), Globals.maxlen, len(chars)), dtype=np.bool)
            y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):

                    X[i, t, char_indices[char]] = 1
                y[i, char_indices[next_chars[i]]] = 1

            return X, y
        X, y = vectorize()

        return Data(
                text = text,
                chars = chars,
                char_indices = char_indices,
                indices_char = indices_char,
                sentences = sentences,
                next_chars = next_chars,
                X = X,
                y = y
                )

class Generate():

    def __init__(self, data):
        self.data  = data

    @staticmethod
    def sample(preds, temperature=1.0):

        """
        helper function to sample an index from a probability array

        preds = e ^(log(preds) / temp) / sum (exp_preds)

        ARGMAX multinomial probas

        preds ->
        temperature ->

        int
        """

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    @staticmethod
    def create_model(chars):

        """


        Creates a Sequential LSTM Model with Dense, and Activation layers

        chars set<chars> -> Model


        """


        model = Sequential()
        model.add(LSTM(128, input_shape=(Globals.maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model

    def train_model(self):

        """

        Trains the model and prints the predictions to standard out.

        """

        model = Generate.create_model(self.data.chars)

        for iteration in range(1,60):

            model.fit(self.data.X,self.data.y, batch_size = 128, epochs = Globals.epochs)
            start_index = random.randint(0, len(self.data.text) - Globals.maxlen - 1)

            for diversity in [0.2, 0.5, 1.0, 1.2]:

                print()
                print('----- diversity:', diversity)

                generated = ''
                sentence = self.data.text[start_index: start_index + Globals.maxlen]
                generated += sentence

                sys.stdout.write(generated)

                for i in range(400):

                    x = np.zeros((1, Globals.maxlen, len(self.data.chars)))
                    for t, char in enumerate(sentence):
                        x[0, t, self.data.char_indices[char]] = 1.

                    preds = model.predict(x, verbose=0)[0]
                    next_index = Generate.sample(preds, diversity)
                    next_char = self.data.indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()

if __name__ == "__main__":

    d = Data.generate('nietzsche.txt')
    gen = Generate(d)
    gen.train_model()

