# 100 Days of ML

Hope to track progress, and encourage myself to keep some of these different projects moving.

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
