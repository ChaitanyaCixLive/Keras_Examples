# -*- coding: utf-8 -*-

'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
'''

from keras.models import Sequential
from keras import layers
import numpy as np

from sklearn.model_selection import train_test_split

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class Globals:

    training_size = 50000
    digits = 3
    invert = True
    max_len = digits + 1 + digits

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class Data():

    chars = '0123456789+ '
    ctable = CharacterTable(chars)

    def __init__(self):

        questions, expected = Data._generate_questions()

        x,y = Data._vectorize(questions, expected)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

        self.x = x
        self.y = y
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val

    @staticmethod
    def _generate_questions():

        numbers = list('0123456789')

        questions = []
        expected = []
        seen = set()

        while len(questions) < Globals.training_size:

            f = lambda: int(''.join(np.random.choice(numbers)
                            for i in range(np.random.randint(1, Globals.digits + 1))))
            a, b = f(), f()
            # Skip any addition questions we've already seen
            # Also skip any such that x+Y == Y+x (hence the sorting).
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            # Pad the data with spaces such that it is always MAxLEN.
            q = '{}+{}'.format(a, b)
            query = q + ' ' * (Globals.max_len - len(q))
            ans = str(a + b)
            # Answers can be of maximum size DIGITS + 1.
            ans += ' ' * (Globals.digits+ 1 - len(ans))
            if Globals.invert:
                # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
                # space used for padding.)
                query = query[::-1]
            questions.append(query)
            expected.append(ans)

        return questions, expected

    @staticmethod
    def _vectorize(questions, expected):
        print('Vectorization...')
        x = np.zeros((len(questions), Globals.max_len, len(Data.chars)), dtype=np.bool)
        y = np.zeros((len(questions), Globals.digits + 1, len(Data.chars)), dtype=np.bool)
        for i, sentence in enumerate(questions):
            x[i] = Data.ctable.encode(sentence, Globals.max_len)
        for i, sentence in enumerate(expected):
            y[i] = Data.ctable.encode(sentence, Globals.digits + 1)

        # Shuffle (x, y) in unison as the later parts of x will almost all be larger
        # digits.

        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        return x,y

class Model():

    def __init__(self,data, rnn = layers.LSTM, hidden_size = 128, batch_size = 128, layers = 1):

        self.data = data

        self.rnn = rnn
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers= layers


    @staticmethod
    def _create_output_str(correct, guess, q):

        if correct == guess:
            correct_str = colors.ok + '☑' + colors.close
        else:
            correct_str = colors.fail + '☒' + colors.close

        strs = [
            f'Q {q[::-1] if Globals.invert else q}',
            f'T {correct}',
            correct_str,
            guess,
            '---'
        ]

        return '\n'.join(strs)

    def _build_model(self):

        print('Build model...')
        model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
        # Note: In a situation where your input sequences have a variable length,
        # use input_shape=(None, num_feature).
        model.add(
                self.rnn(self.hidden_size, input_shape=(Globals.max_len, len(Data.chars)))
                )
        # As the decoder RNN's input, repeatedly provide with the last hidden state of
        # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
        # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
        model.add(layers.RepeatVector(Globals.digits+ 1))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(self.layers):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            model.add(self.rnn(self.hidden_size, return_sequences=True))

        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.

        model.add(layers.TimeDistributed(layers.Dense(len(Data.chars))))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        return model

    def run(self):

        model = self._build_model()

        for _ in range(200):

           model.fit(self.data.x_train, self.data.y_train,
                      batch_size=self.batch_size,
                      epochs=1,
                      validation_data=(self.data.x_val, self.data.y_val))

           for _ in range(10):
               ind = np.random.randint(0, len(self.data.x_val))
               rowx, rowy = self.data.x_val[np.array([ind])], self.data.y_val[np.array([ind])]
               preds = model.predict_classes(rowx, verbose=0)
               q = Data.ctable.decode(rowx[0])
               correct = Data.ctable.decode(rowy[0])
               guess = Data.ctable.decode(preds[0], calc_argmax=False)
               print (Model._create_output_str(correct, guess, q))

if __name__ == "__main__":

    data = Data()
    model = Model(data)
    model.run()

