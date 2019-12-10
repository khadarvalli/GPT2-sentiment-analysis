![image](https://openai.com/content/images/2019/03/openai-about.jpg)
# GPT-2 for Sentiment Analysis
What you'll do with this paper:
* Retrain OpenAI's GPT-2 model on Sentiment Analysis
* Use it if you want
* That's it bro <br>

WaIt WhAtS sEnTiMeNt AnAlYsIs? <br>
Computer says a certain phrase means good or bad thing. <br>
### What's GPT-2?
GPT-2 Is what's called a transformer model but I won't bore you with the boring stuff, if you want more information the here's the paper about it from OpenAI: https://openai.com/blog/better-language-models/ <br><br>

Either you can retrain this thing yourself or just use my code [here](https://colab.research.google.com/drive/1ulO-Z0G6BdvQAZ83PJNXCS0ygtRtp46g) <br>
I recommend doing this in a [Google Colab](https://colab.research.google.com) or [Jupyter Notebook](https://jupyter.org/), it makes installing dependencies easier. <br>
Otherwise every line of code with a 
`!`
in the front you should run in the terminal. <br><br>

    %tensorflow_version 1.x # Only use this line if you're in google colab, otherwise make sure the tensorflow version is less than 2
    !pip install numpy
    !pip install tensorflow
    !pip install gpt-2-simple 
    import json
    import os
    import numpy as np
    import tensorflow as tf
    import gpt_2_simple as gpt2
    
    model_name = "345M" # The GPT-2 model we're using
    
    gpt2.download_gpt2(model_name=model_name) # Download the model`
The first four lines are just installing the things we need.
and then importing them.
The `model_name	` variable is the GPT-2 Model we're using, this is the medium size model because any bigger is too hard to train.
The last line is just downloading the model so we can retrain/finetune it.

Now that we have everything initialized, we just need the dataset to retrain it on.

    !wget https://github.com/spronkoid/GPT2-sentiment-analysis/blob/master/sentiment50.txt
    
    file_name = "sentiment50.txt"

This is a dataset I compiled from 50 positive and 50 negative reviews from the Stanford movie review dataset.

Now we can retrain the model.

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
    file_name,
    model_name=model_name,
    steps=400)  # steps is max number of training steps
I would recommend letting this train overnight, as you can just sleep the time away that it uses. Because of this, I have no clue how long it takes, though 250 steps took around 6 hours and 9 minutes. 

Cool! Now we have a trained model we just need to be able to interact with. So I took the official way to interact with it and changed it a bit for us

    def interact_model(
        model_name,
        seed,
        nsamples,
        batch_size,
        length,
        temperature,
        top_k,
        models_dir
    ):
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0
    
        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
    
        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k
            )
    
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)
    
            while True:
                raw_text = input("Model prompt >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("Model prompt >>> ")
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(text)
                print("=" * 80)


Epic! Now there's some dependencies that we need to install for this to work

    !git clone https://github.com/openai/gpt-2.git
    import os
    os.chdir("gpt-2/src/")
    import model, sample, encoder
    os.chdir('../../..')

Now we can interact with it like this:

    interact_model(
        'run1',
        None,
        1,
        1,
        2,
        1,
        0,
        './checkpoint'
    )


It will show some warnings but it's okay.
When interacting with this, we have to put this `//` in front of the phrase and `||` after it. Why? Because that's how I formatted the data to denote to the model when a phrase starts and stops, otherwise it would just keep generating more of a prompt.


It seems to perform well with average sized reviews of things, but I've gotten weird results with super short and super long prompts. Though this is probably understandable.

When giving it a prompt is should respond either Positive or Negative, for some reason they aren't capitalized all the time? I have no clue why. But like ahaaaaaaaaaaaaaaaaaaaaaaaaaa we're done your welcome.

by [@spronkoid](https://github.com/spronkoid)
