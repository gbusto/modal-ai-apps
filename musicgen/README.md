# MusicGen Modal App

This is a Modal app for generating music using the MusicGen model. I wasn't able to find any online music generation APIs, so I created my own simple one.

You can read more about musicgen and its models [here](https://musicgenai.org/musicgen-models/).

## How to use this

1. Create an account on [Modal](https://modal.com/) and add your billing info
1. Clone this whole project repository
1. `cd musicgen` to change into this directory
1. `cp .env.example .env` to create an `.env` file
1. Run `modal setup` to set up the Modal CLI and auth with your account
1. Create the `models` volume by running: `modal volume create models`
1. `./deploy_musicgen.sh` to deploy the app to Modal
1. Take the resulting URL for your deployment and put it in the `.env` file as `MODAL_APP_URL`
1. Run `export $(cat .env)` to set the environment variables
1. Run the client to generate music:

```bash
# Download the model to your Modal volume first
python client.py download --model facebook/musicgen-small

# You can also download medium and/or large models (these will all take ~1-2 minutes)
python client.py download --model facebook/musicgen-medium
python client.py download --model facebook/musicgen-large

# Then generate music with a prompt, duration, and a specific model
python client.py generate "Strong orchestral score; Hans Zimmer style" --duration 15 --model facebook/musicgen-medium
```

It will take 30-80 seconds in total to generate music, and the script will receive the music and save it to `output.wav`. The smaller model is faster, but lower quality. The larger model takes a bit longer (closer to 80 seconds), but the quality is supposed to be better.

I haven't generated enough music yet, but I don't really hear a huge difference in quality between the models.

Since the generation can take a while, it will make the API request to start the generation, then check the status every 10 seconds until it's complete.

We have parameters in the code to keep the container alive for 60 seconds, so it will keep running even if the API is idle in order to make subsequent generations a little faster.

If you run into issues, please let me know!

## What is Modal?

Modal is a platform for running Python code in the cloud. Great for ML / AI scripts as it lets you pick and run your code on GPUs, CPUs, or both. You can configure the resources you need by function, and it's billed by the second.

You can learn more about it [here](https://modal.com/), or read [this longer description](https://www.perplexity.ai/search/how-would-you-describe-what-mo-yz.2mCMLTvy3kQkO_JOdAA) created by Perplexity when I asked it to describe the Modal platform.

## Future Improvements
I should take more time to understand this code, check versions, etc. I'll admit I was a little over my skis relying on AI for the CUDA stuff, but I understood and worked through a lot of the errors that popped up to get it to this point.

- [ ] Support some kind of authentication header to prevent abuse
- [ ] Potentially support other ouptut formats
- [ ] Allow the user to name the output file and specify an output directory
- [ ] Add support for the Facebook MusicGen Melody and Stereo models if needed
