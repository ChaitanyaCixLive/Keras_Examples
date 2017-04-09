# the world at large can ignore this
import os

def read_data():

    init_dir = '../breibart_data/articles'


    acc = []

    for filename in os.listdir(init_dir):

        name = init_dir + "/" + filename

        with open(name, 'r') as f:

            acc.append(f.read())

    return ''.join(acc)
