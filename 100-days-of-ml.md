# 100 Days of ML

Hope to track progress, and encourage myself to keep some of these different projects moving.

### 8 September
Career stuff and some reading

### 7 September
- Some interview prep
- I want to train a GPT-2 for Arabic, including dialect control tokens. I've created the tokenizer, but the next steps are confusing, seems we are only replacing tokens from English GPT-2 , where does it do the model training?? Does everything need to be finetuned?

### 6 September
- Some practice questions for tech screening interviews
- PRs to AdapterHub adding Arabic dialect task, issues with GPT-2 tutorial, RTL layout for Arabic translation of PyTorch DL book

### 5 September
Had some trouble sleeping, most progress was running LandCoverNet dataset download
- Wrote down a plan for AdapterHub, would be interesting to make Arabic dialect and COVID MASK adapters

### 4 September
- Followed the directions from Facebook for the HateMemes challenge ( https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes ), got a few epochs of training, but it would take >24hrs of training
- Helped blog post go into a publication, The Startup (not sure whether using publications benefits me?)

### 3 September
- I finish my MASK-ing demo for BERT. Many words are successfully inserted into the model, but (a) the coefficients that I used seem arbitrary, and (b) overfitting is severe (Wuhan suggested for generic city name)
- Blog post: https://medium.com/@mapmeld/patching-pre-trained-language-models-28ed6ea8b0bc
- I reran HateMemes AutoKeras testing to investigate why I dropped it months ago. GPU can only handle around 200x200 pixel images. Accuracy metric indicates it is only predicting one category that is >50% of the training set. Need to use TPU? And multi-modal?
- Transformers had a new release recently, and Ktrain updated their dependency. Issue closed.

### 2 September
Still not tech progress - I will have more time now that my assigned work-days have ended for the long weekend.
- Watched longform videos, Cultivating ML Communities https://www.youtube.com/watch?v=uKjX-iJGKyA, Toxic Language on RASA, Getting Started in Automated Game Design https://www.youtube.com/watch?v=dZv-vRrnHDA

### 1 September
Not a lot of tech progress today. I should write a script to collect sentences for the MASK project from Wikipedia and Reddit, since that will take time to process.
- Read up on fairness/consent around data collection and archiving https://arxiv.org/abs/1912.10389
- Re-downloaded Hate Memes challenge data so I can do some experiments in the future
- JAX: in addition to Objax, there's a framework called Elegy which might be better

### 31 August

Notebook: https://colab.research.google.com/drive/1lQfWFOPZCO5IVVw_3YwiXEY96fZ1zhoV

- Found BERT and T5's suggested words for face mask, distancing, etc. Can calculate numeric weights for BERT, but not T5
- Coded ways to insert new words into tokenizer and their vectors into the transformer. Raises a warning but should be OK.
- I can overfit an embedding with torch.mean that comes up first in the suggested responses. But it seems better to find 100s of organic examples and fit from that
- I propose creating a benchmark of coronavirus-relevance by 2020 sentences ("social distancing") and old context sentences ("social atmosphere") in case a good model can distinguish

### 30 August

- Used Tokenizers to make vocabulary lists from 15k to 40k words/word-pieces. Some contained 'coronavirus' or '-avirus'. Compared vocabulary lists to BERT and T5
- Unfortunately 3,000+ tokens were new (e.g. typewriter as one token). I tweaked tokenizers settings with smaller vocabulary sizes, reduced to ~300 new tokens. But I didn't find a sweet spot of # of tokens to include coronavirus and not these additional pieces.

### 29 August

- Started writing blog post and testing Tokenizers to set up my patching language models post
- Started using WikiExtractor and Tokenizers to find popular new words/tokens in English Wikipedia. The English version is considerably larger than what I've worked with before, so eventually I had the file in Google Drive, unzipped there (bzip2 -d), then copied unzipped XML to CoLab
- Found a recent Facebook research preprint (https://arxiv.org/abs/1910.06241) which has a similar idea around updating pretrained models. They made their own simulated corpuses using fastText, and not updating a massive, real-world, transformers-based model, so I feel good about this.

### 28 August (and recent)

- Resumed training of Hindi-TPU-Electra (base-sized model). The last training update under-performed compared to previous versions of this model, but I wanted to continue training and try again. Loss vs. learning rate is proving confusing in ktrain
- ktrain cannot load ELECTRA models - turned out to be an issue in current pip release of Transformers; until that gets updated, you can pip install Transformers from source. Filed an issue on ktrain with what I learned
- First attempt at Bangla-Electra + Longformer; made edits from RoBERTa input model -> ELECTRA AutoModel + AutoTokenizer input model. Got errors due to vector shape mismatch, I assume this is because output is RoBERTa and input is no longer RoBERTa? Can the two be compared? Or would it be better to train longformer from scratch
  Notebook: https://colab.research.google.com/drive/1QhddQxz9DvFa-jVJTgO19CrWbYcfJswK
- Recently added BERT [mask] widget to HuggingFace models, but it usually returns the same unhelpful words? Might not be a fit
- Made an OpenStreetMap screencast using the RapiD AI-assisted map, need to try again with RapiD+Esri tool
- Heard about Objax - https://github.com/google/objax - new ML framework based on JAX, this might be a good fit for the LandCoverNet dataset, but reluctant to do many new things at once
