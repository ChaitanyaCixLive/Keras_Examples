# Text Generator


I was going through the Keras examples and I found the text generator one. I have alot of text data so it seemed like a cool idea to try it out. However,
I didn't like how it was written so I rewrote it to better suite my style and so that I understood it better.

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

## How to Tweak

Change the global values struct to modify parameters


