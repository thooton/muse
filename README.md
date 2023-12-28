# muse

Muse is a Python script for creating synthetic textbooks using Google's Gemini Pro API, providing free access at 60 req/min. On average, each request generates ~1000 tokens, resulting in 3.6 million tokens per API key per hour!

## Usage Instructions

1. Clone or download the repository to your local machine.
2. Enter the directory.
3. Run `python3 -m pip install -r requirements.txt`.
4. Run `python3 main.py` to generate a `config.ini` file with default values.
5. Edit `config.ini` and add one or more API keys obtained from [Google's Makersuite](https://makersuite.google.com/app/apikey). Feel free to make other adjustments as well.
6. On Hugging Face, go to [New Dataset](https://huggingface.co/new-dataset) and create a new dataset named `muse_textbooks`.
7. Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens) to get your current access token (API key) or create a new one with **write** permission.
8. Run `python3 main.py`; you'll be prompted to enter your Hugging Face API key.

The script submits 60 requests/minute to Google's Gemini Pro, generating textbooks in the specified (`out_dir`) directory and auto-uploading them to a Hugging Face dataset named `your_username/muse_textbooks`.

**Note**: Configuration values can be specified either in the config.ini file or passed as environment variables. It is advisable to avoid setting arbitrary values for temperature and top_p.

### Nix/Nixos

Run: `nix-shell` inside the muse directory. Then follow instructions from the above-mentioned step 5.

## Implications

By creating large amounts of open-source synthetic textbook data, we pave the way for open-source models that are more efficient and performant. phi-2 was trained on 250B tokens of mixed synthetic data and webtext; what might we be able to do with a 7B model trained on trillions of synthetic tokens?

## How it works

The program auto-generates prompts by sampling from two seed datasets defined by the `TEXT_DATASET` and `CODE_DATASET` variables in `data_processing.py`. After a passage is sampled, it is passed to one of three prompt templates defined by `TEMPLATES` that instruct the LLM to either:
1) Generate a two-person debate on a subject related to the passage, or
2) Generate an informative lecture on a subject related to the passage, or
3) Generate a computer science textbook section on a subject related to the passage.

## Contributing

This repository is open to any PRs/issues/suggestions :)
