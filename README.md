# muse
Muse is a simple python script for creating synthetic textbooks using Google's Gemini Pro API, which is currently providing free access at 60req/min. At current rates it generates about 1.2M synthetic tokens every 15 minutes.

To get started, simply clone or download the repository to your local machine, enter the directory, and run `python3 -m pip install -r requirements.txt`. Open `index.py` and replace `GEMINI_API_KEY` with an API key obtained from https://makersuite.google.com/app/apikey.

After you run `python3 index.py`, you'll be prompted to enter in your HuggingFace API key. The script will generate textbooks in the `./textbooks` directory and auto-upload them to a HuggingFace dataset named `your_username/muse_textbooks`.

By creating large amounts of open-source synthetic textbook data, we pave the way for open-source models that are far more efficient and performant. phi-2 was trained on 250B tokens of mixed synthetic data and webtext; what might we be able to do with a 7B model trained on trillions of synthetic tokens?

This repo is open to any PRs/issues/suggestions :)
