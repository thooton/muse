# About

Muse is a simple python script for creating synthetic textbooks using Google's Gemini Pro API, which is currently providing free access at 60 req/min. On average, each request generates ~1000 tokens. This results in 3.6 million tokens per participant per hour!

## Usage Instructions

1. Clone or download the repository to your local machine.
2. Enter the directory.
3. Run `python3 -m pip install -r requirements.txt`.
4. Run `python3 main.py` to generate a config.ini file with default values.
5. Edit `config.ini`, and add your API key obtained from https://makersuite.google.com/app/apikey. Feel free to make other adjustments as well.
6. On huggingface, go to https://huggingface.co/new-dataset and create a new dataset named `muse_textbooks`.
7. Go to https://huggingface.co/settings/tokens to get your current access token (API key) or create a new one. Make sure it has **write** permission.
8. Run `python3 main.py`, you'll be prompted to enter in your HuggingFace API key.

And that's it! The code will submit 60 requests/minute to Google's Gemini Pro. The script will generate textbooks in the specified (`out_dir`) directory and auto-upload them to a HuggingFace dataset named `your_username/muse_textbooks`.

### Nix/Nixos

Run: `nix-shell` inside muse directory. Then follow instructions from above mentioned step 5.

## Implications

By creating large amounts of open-source synthetic textbook data, we pave the way for open-source models that are far more efficient and performant. phi-2 was trained on 250B tokens of mixed synthetic data and webtext; what might we be able to do with a 7B model trained on trillions of synthetic tokens?

## How it works

The script autogenerates prompts by sampling from two seed datasets which are defined by the `TEXT_DATASET` and `CODE_DATASET` variables in `main.py`. After a passage is sampled, it is passed to one of three prompt templates defined by `TEMPLATES` that instruct the LLM to either 1) generate a two-person debate on a subject related to the passage, 2) generate an informative lecture on a subject related to the passage, or 3) generate a computer science textbook section on a subject related to the passage. All of this can of course be tweaked by editing `main.py`.

## Contributing

This repository is open to any PRs/issues/suggestions :)
