# LSTM Text Generator
Feed the text generator of text file. Alternatively use process.py to feed it a folder with files of .txt and it will concatenate them into a large string.

The output will be new string generated from sample the space of in the input text.

## How to Use

```python
filename = "50shadesofgray.txt"

# reads text file and generates features from it

data = Data.generate(filename)

# Initializes model

gen = Generate(data)

# Runs model
gen.train_model()
```

# Addition RNN

```python

# generates input data 

data = Data()

# Initializes the model
model = Model(data)

# Runs model
model.run()

```






