# GPT2-sentiment-analysis
OpenAi's GPT-2 is a Transformer model blah blah blah... <br>
[Medium Article here](https://link.medium.com/Ux2lA4S8h2) <br>
Here's how you train it.

 1. Download as zip
 2. Just run the stuff in Jupyter or Google Collab (or use [this collab link](https://colab.research.google.com/drive/1ulO-Z0G6BdvQAZ83PJNXCS0ygtRtp46g), where I have everything set up. )
<br>
 3. Yeah that's it.

(update: using max's new code https://github.com/minimaxir/aitextgen it's way faster, and using the smaller model is probably better for this small task, just 2000 iterations and it's at 0.03 loss, pretty fucking good if you ask me)

Now that it's set up, you should format your prompts like this:

    // your prompt here ||
    
Then the model should return either Positive or Negative, though I've seen some weird results with super short and super long prompts that don't make sense.
<br>
You can zip your model with (helpful for collab when downloading the model):

    import shutil
    
    shutil.make_archive("model",  'zip',  "/content/checkpoint")

And then download the zip.


### Details

The GPT-2 Model by OpenAI is detailed [here](https://openai.com/blog/better-language-models/). <br>

Why didn't I use the full model? Because there's no way that my average consumer laptop could train that, or most computers to be honest. <br>

It's trained on 50 Positive and 50 Negative samples from the Stanford movie reviews data-set. Which I had to hand write because my script to just do it automatically to make a bigger set didn't work and I wasn't bouta waste four hours on a script that'll do something that I'll do in one. (that's the sentiment50.txt in the files). <br>
For usage rights refer to license.txt

by [@spronkoid](https://twitter.com/spronkoid)
