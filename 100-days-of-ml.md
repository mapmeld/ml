# 100 Days of ML

Hope to track progress, and encourage myself to keep some of these different projects moving.

### 12 October
- Ran updated Bangla/Bengali benchmarks with new model, reached out to the developer
- Asked Maldivian devs about adding sample sentences

### 9-11 October
- Read papers on arxiv, contract work, not tracking / little progress

### 8 October
- read about ar/en gigabert

### 7 October
- Submitted breakout session to neurips workshop

### 6 October
- Can't remember making progress on this day

### 5 October
- Figured out issues with Elegy: the library is changing and still needs to be documented
- Watched TWIML video about Duolingo
- Contract work

### 4 October
- Wrote JAX post, I think I resolved flowers on Flax/Linen today too

### 3 October
- Completed Imagenette examples, some flower examples, no luck on Elegy
- Breakout session write-up

### 2 October
- Advocating for, and researching re: NeurIPS breakout session
- Loaded Imagenet example from Haiku and ported to Imagenette

### 1 October
- Watched "NLP with Friends", reading into WinoWhy and HKUST's ASER knowledge graph
- Set up contract work repo
- Exploring Flax/Linen after it was recommended by dev on Twitter

### 30 September
- Watched "GANs for Good" seminar
- Submitted 2-minute talk on WEAT-ES to TWIML lightning round

### 29 September
Retried Flax parsing dataset... will likely scrap my code, use workig Imagenet example, and adapt from there

### 28 September
Slower day, read some papers, FaccT videos

### 27 September
- Confirmed AutoKeras bug is fixed in upcoming release; updated + simplified AutoKeras flowers example
- Started Flax version of flowers example based on Imagenet - this seems like a bulk of code, initially batches are not quite fitting expected iterator.. confirmed TFDataset is hard to input https://twitter.com/wightmanr/status/1302039882575917057
- AI Ethics post in "The Startup" Medium publicatio

### 26 September
- Watched OpenMined privacy conference
- Finished post on AI Ethics

### 25 September
- Added eval run to Kaggle flowers + Objax, not yet going to a new framework. Proposed talk on this
- Read papers including this on differential privacy - DP models increase bias/inaccuracy on minority classes https://arxiv.org/pdf/2009.06389.pdf

### 24 September
- Frustration with LandCoverNet seems to be common https://twitter.com/MatthewTeschke/status/1306370474415456258
- Watched RASA video on NLP CheckList paper
- Finished getting Kaggle flowers example running in Objax (training only, not testing) - want to try Flax and other JAX stuff

### 23 September
- Watched Andrew Ng webinar and Chai Time Data Science
- Arabic dialect adapter got merged, received feedback for improvement (Farasa segmentation before tokenization), redid model and fixed bug in Gulf dialect input
- Finished write-up on moving target metric

### 22 September
- Watched UCI Symposium on Reproducibility, video on AI for NGOs https://www.youtube.com/watch?v=Y0dKDcipaHY
- Uploaded adapter for Arabic dialects on AdapterHub
- Implementing my proposal for moving target NLP metric

### 21 September
- Rebuilt multi-Arabic dialect dataset, re-trained Sanaa-Dialect (finetuned Arabic GPT-2) on this larger dataset
- Watched ML videos

### 20 September
Minimal progress, participated in startup meetup

### 19 September
- Watched videos / lesson plans on Teaching ML https://www.youtube.com/channel/UCJWzJlUp-iZtCMhsohvExPA
- Read through selected projects on Mozilla Trustworthy AI
- Longer run of GPT-2 on larger Arabic dataset

### 18 September
Traveling, minimal reading about NLP Checklist, published blog about Arabic GPT-2 work

### 17 September
- Made interactive/explorer for the refugee data project. 
- I liked this question about how to have a class of newbies do something interesting with neural nets including internals https://twitter.com/GalaxyKate/status/1306686937256865792
- Built a new Arabic wiki dataset (2-3x previous) to train GPT-2 model, running process overnight

### 16 September
Joined AI ethics meetup, collected sources and made a visualization for a refugee data project (but it is ~500-2,000 records / dataset, so not ML)

### 15 September
- Arabic GPT-2 can generate text, but finetuning in SimpleTransformers ran into a weird bug. I tried an older release of SimpleTransformers which introduced new issues, until I found a GitHub issue matching my problem. The next issue was CUDA/GPU running out of space. Every time the GPU gets messed up, have to hit factory reset on everything. Had to reduce training batch size and cut some other corners to fit.
- Ultimately got a dialect model which I could upload on its own, and link as an example for sanaa. I plan to retrain this on more data.

### 14 September
- Tried flowers/AutoKeras code on Kaggle notebook. There's the same RAM limit of about 4000 192x192px images. I could see if there's a better way to process the images in batches, or lose a layer of abstraction and use Keras-Tuner (supports TPU, which AutoKeras can't). 
- Uploaded a basic Arabic GPT-2 ('sanaa')
- AdapterHub asked me to upload an example for the dialect detector. I'm on the right track; current issue is rebuilding my data, and turning my CSVs into torch.utils.data.Dataset format, so I can get the Transformers Trainer to process it

### 13 September
No real progress, watched a video about Kaggle.

### 12 September
- I confirmed that AutoKeras has an 84% accuracy score on the flowers dataset. This was impressive because I hadn't had the RAM to use the whole dataset, and I used only the CoLab GPU. I filed an issue asking AutoKeras to support the dataset (I think the issue is that x and y are included)
- I evaluated Elegy and Objax as libraries for a comparable JAX-based image classifier. Objax looks better supported, and has examples using ResNet and CGIAR. I figured out how to load the dataset and transpose the image's dimensions to match expected input. Objax models look something like PyTorch code. Unfortunately I still have a dimensions mismatch in the loss function. The right thing might be to run their ImageNet example and track data shape through these same functions.

### 11 September
- The new tutorial created a small but productive Arabic text-generation model. The text output isn't so meaningful, but I might be able to continue training on the tail of the Wikipedia articles. I might be able to engineer the TensorFlow model into a standardized tokenizer+transformer for HuggingFace. The end goal is still adding tokens to control which dialect is output.
- I found an images dataset which is more straightforward (Kaggle flowers/TPU dataset). The data is distributed as TFRecords (x and y bundled together). Based on code samples and a Kaggle notebook, I was able to read in TFRecords, then separate out x and y to feed them into AutoKeras. CoLab RAM cannot handle 10,000 small images. I was able to run with 4,000 images, and there are 104 classes of flowers. I saw 72% validation accuracy here. But I should prove this on a new dataset. If it's real, it could be interesting then to show this problem in JAX.

### 10 September
Making another attempt at Arabic GPT-2 (different tutorial as basis) and LandCoverNet (reference notebook got updated)

### 8-9 September
Career stuff and some reading, stop of progress

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
