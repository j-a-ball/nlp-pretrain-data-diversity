# How does pretrain dataset diversity relate to a pretrained language model's performance after finetuning?

## Course project for Stanford CS224U: Natural Language Understanding.

Thanks to Chris Potts and Aasavari Kakne for teaching support, and Douwe Kiela for the project idea.

## Metrics 
([Palumbo et al., 2020](https://bit.ly/3cVfx3x); [Stasaski et al., 2020](https://aclanthology.org/2020.acl-main.446/))

The following four metrics were successfully scaled to a subset of the [Common Crawl News corpus](https://huggingface.co/datasets/cc_news) containing more than 700,000 English news articles:

  1. Mean-IDF (Baeza-Yates et al., 1999)
      - selecting documents for pretraining with the highest [tf-idf scores](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), averaged over all terms in each document and computed using [scikit-learn's TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  2. Trigram language model entropy
     - selecting documents with the highest trigram entropy scores, computed for each document using [NLTK's Language Modeling Module](https://www.nltk.org/api/nltk.lm.html)
  3. "Outlier distance," or Euclidean distance from mean corpus vector (Stasaski et al., 2020; Larson et al., 2019)
     - first point-wise averaging of each document's BERT word vectors, then selecting docs with greatest Euclidean distance from the average of all BERT word vectors in the corpus
  4. "Word Embedding Diversity (WED)," or mean pairwise cosine distance (Palumbo et al., 2020)
     - first point-wise averaging of each document's BERT word vectors, then selecting docs with the greatest average cosine distance, computed pairwise between each doc and all other docs in the corpus
    
The following two metrics were not successfully scaled to the entire corpus, either due to being too inefficient or too memory-intensive at n > 700,000:

  5. Jaccard distance ([scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html))
  6. Self-BLEU (Zhang et al., 2017; Alihosseini et al., 2019) ([github repo](https://github.com/Danial-Alh/fast-bleu))
  
An additional, 7th diversity metric was proposed by [Lai et al., 2020](https://aclanthology.org/2020.lrec-1.215/), computed as the geometric average of standard deviations along each axis in an embedding space for any number of documents. Lai et al.'s paper is an excellent contribution, and I plan to reformulate this experiment to test only the metrics they propose.
    
## Experimental design:

  5 separate ["bert-base-uncased" models](https://huggingface.co/bert-base-uncased) were additionally pretrained on selections of 10,000 cc_news articles each. The first model was trained on 10,000 articles selected at random. The remaining four models were trained on samples of 10,000 cc_news articles each, selected to maximize mean-IDF, trigram entropy, outlier distance, and word embedding diversity scores. Each pretrained model was then finetuned and evaluated on the Stanford Question Answering Dataset v1.1 ([SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/)).
  
  The masked language modeling script used for additional pretraining can be accessed [here](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py), and the SQuAD finetuning script can be accessed [here](https://github.com/huggingface/transformers/blob/main/examples/legacy/question-answering/run_squad.py). Default parameters were not altered, save the maximum sequence length for each article, which was set to 512 (i.e., a maximum of 512 BERT WordPiece tokens for each article).
  
## Results:

  The separately trained BERT language models did not vary significantly in their finetuned SQuAD performance. The model pretrained on articles selected according to trigram language model entropy performed best, although this finding is probably not significant. It seems that the additional 10,000 news articles did not provide enough opportunity for the models to learn different behavior. This suggests that the impacts of dataset diversity would only be felt (a) when data are few, which is not normally the case for pretraining, or (b) when the diversity-selected dataset is a couple orders of magnitude larger than a mere 10,000 short news articles.
  
  Two heuristics developed from this project:
  
  1. Pretrain datasets for NLP models at the scale of BERT should consist of 100M words or more.
  2. Simpler language models, such as trigram language models, are well suited to the task of selecting data inputs for more complex language models.

Please reach out to me at {jonball -at- stanford -dot- edu} if you would like to try scaling up this project!
