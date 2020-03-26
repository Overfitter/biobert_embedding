# BioBert Embeddings
Token and sentence level embeddings from BioBERT model (Biomedical Domain).

[BERT](https://arxiv.org/abs/1810.04805), published by Google, is conceptually simple and empirically powerful as it obtained state-of-the-art results on eleven natural language processing tasks.  

The objective of this project is to obtain the word or sentence embeddings from [BioBERT](https://github.com/dmis-lab/biobert), pre-trained model by DMIS-lab. BioBERT, which is a BERT language model further trained on PubMed articles for adapting biomedical domain.

Instead of building and do fine-tuning for an end-to-end NLP model, You can directly utilize word embeddings from Biomedical BERT to build NLP models for various downstream tasks eg. Biomedical text classification, Text clustering, Extractive summarization or Entity extraction etc.



## Features
* Creates an abstraction to remove dealing with inferencing pre-trained BioBERT model.
* Require only two lines of code to get sentence/token-level encoding for a text sentence.
* The package takes care of OOVs (out of vocabulary) inherently.
* Downloads and installs BioBERT pre-trained model (first initialization, usage in next section).

## Install
```
pip install biobert-embedding==0.1.2
```

## Examples

word embeddings generated are list of 768 dimensional embeddings for each word. <br>
sentence embedding generated is 768 dimensional embedding.

```python
from biobert_embedding.embedding import BiobertEmbedding

## Example 1
text = "Breast cancers with HER2 amplification have a higher risk of CNS metastasis and poorer prognosis."\

# Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
biobert = BiobertEmbedding()

word_embeddings = biobert.word_vector(text)
sentence_embedding = biobert.sentence_vector(text)

print("Text Tokens: ", biobert.tokens)
# Text Tokens:  ['breast', 'cancers', 'with', 'her2', 'amplification', 'have', 'a', 'higher', 'risk', 'of', 'cns', 'metastasis', 'and', 'poorer', 'prognosis', '.']

print ('Shape of Word Embeddings: %d x %d' % (len(word_embeddings), len(word_embeddings[0])))
# Shape of Word Embeddings: 16 x 768

print("Shape of Sentence Embedding = ",len(sentence_embedding))
# Shape of Sentence Embedding =  768

## Example 2
sentence_vector1 = biobert.sentence_vector('Breast cancers with HER2 amplification have a higher risk of CNS metastasis and poorer prognosis.')
sentence_vector2 = biobert.sentence_vector('Breast cancers with HER2 amplification are more aggressive, have a higher risk of CNS metastasis, and poorer prognosis.')

cosine_sim = 1 - distance.cosine(sentence_vector1, sentence_vector2)
print('cosine similarity:', cosine_sim)
#cosine similarity: 0.992756187915802
```
