## Abstract
This research study investigates the efficacy of various Natural Language Processing (NLP) models for emotion analysis on a diverse corpus of textual data. The objective is to discern the comparative performance of traditional feature extraction methods, including Bag-of-Words (BoW), TF-IDF, and N-grams in conjunction with Support Vector Machines (SVM), alongside more advanced techniques such as word embeddings and the state-of-the-art transformer-based model BERT.

Keywords: Emotional analysis, BoW, TF-IDF, N-gram, word embedding, SVM, transformers, BERT

## Introduction

Emotion analysis, often referred to as sentiment analysis or opinion mining, involves the computational examination of text to discern and categorize the emotional tone, opinions, or sentiments expressed within. The goal is to unravel the latent emotional states underlying textual data, providing valuable insights into the subjective aspects of human communication. This not only facilitates a deeper understanding of individual expressions but also opens avenues for applications in diverse fields.

In recent years, there has been a growing recognition of the crucial link between emotion analysis and mental health. The ability to decipher emotional states from textual data offers a non-intrusive and scalable approach to gauging the emotional well-being of individuals. Recognizing patterns of distress, anxiety, or positivity in online communication can contribute to the early identification of mental health concerns, paving the way for timely intervention and support.

This project harnesses advanced NLP techniques for emotion analysis with a specific focus on mental health implications. By leveraging state-of-the-art models and methodologies, the project endeavours to uncover nuanced emotional cues embedded in textual data. The aim is to contribute to the development of innovative tools and methodologies that can assist mental health professionals, researchers, and support systems in gauging and addressing emotional well-being in the digital landscape.



## Dataset
In this study, the dataset was obtained from a [Kaggle Notebook](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp). A collection of .txt files was merged to create our dataset. The dataset contains two columns- sentence, label. Sentence column has unique values and is mapped to a label. The label describes the emotion being conveyed through the corresponding sentence. Label can take values- sadness, anger, love, surprise, fear, joy. The dataset consists of 19999 rows.
The sentence column was further processed. The stop words were removed. WordNet Lemmatizer was used to reduce words to their lemmas. 
The count of different emotions in the dataset was the following-

  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/06db77fa-f8da-4742-a3e4-a8dffa038743)


The word cloud of each emotion after processing of sentences was as follows-

  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/4f658042-5cc6-406d-8b54-686b93d7effe)
  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/c3b470ca-1888-4b94-944e-32a2c6611061)
  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/44b30e81-6c13-41da-80fa-55c203907fcc)
  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/79b3c39c-32e6-4ddd-b76c-7a2d7075b0d8)
  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/42610b17-3a3e-4f0b-8a8e-648488b433f7)
  ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/b2c50704-d4b4-4f0a-8436-41152db76b89)

  
## Models

**Support Vector Machine (SVM):** SVM is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates different classes in the feature space. SVM is effective for high-dimensional data and is widely used in text classification tasks.

**Bag-of-Words (BoW):** BoW is a simple and widely used technique in natural language processing for text representation. It represents a document as an unordered set of words, disregarding grammar, and word order. The frequency of each word in the document is used as a feature for further analysis.

**Term Frequency-Inverse Document Frequency (TF-IDF):** TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. It considers both the frequency of a term in a document and its rarity across the entire document collection, providing a  more nuanced representation than BoW.

**N-gram:** N-grams are contiguous sequences of n items (words, characters, etc.) from a given sample of text or speech. They are used to capture local word patterns and relationships in a document. For example, bigrams (2-grams) represent pairs of consecutive words.

**Word2Vec:** Word2Vec is a word embedding technique that represents words as dense vectors in a continuous vector space. It captures semantic relationships between words and is trained on large corpora to generate word embeddings. Similar words have similar vector representations.

**BERT (Bidirectional Encoder Representations from Transformers):** BERT is a pre-trained transformer-based language model that captures bidirectional contextual information in text. It is widely used for various natural language processing tasks. BERT considers the context of each word by looking at both its left and right surroundings, leading to rich contextualized representations.

## Experimental Setup
The dataset has been processed to remove stop words from sentences and the words have been lemmatized. 
The dataset has been split in 80:20 for training-testing.
The specification for the models are as follows-

a.	**BoW**- Count Vectorizer has been used. 

b.	**TFIDF**- TfidfVectorizer with max_features=5000

c.	**N-gram**- Count vectorizer used with ngram_range=(1,3) and max_features=5000

d.	**Word2vec**- vector_size=100, window=5, min_count=1, sg=0, Train- epochs=10

e.	**BERT**-
Label encoder has been used to encode the 6 emotions.
batch size=16, epochs=3, AdamW, learning rate=2e-5

## Results

| ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/7efd3248-fae1-4afe-8602-556b63438e54) |
| :--: |
| *BoW and SVM* |


| ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/d3ae9cb2-1f74-4309-9a4e-acd7b00dd00b) |
| :--: |
| *TF-IDF and SVM* |

| ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/6fa5d621-6edf-4615-9d3e-dc1588940fda) |
| :--: |
| *N-gram and SVM* |

| ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/05053133-b548-45f2-9fb3-cfb175de94a7) |
| :--: |
| *word2vec and SVM* |

| ![image](https://github.com/portfolio-sid/emotion-analysis/assets/174100427/674e1dd8-93ec-4b4b-a7c4-b2406b2accf1) |
| :--: |
| *BERT* |


**1.	Baseline Feature Extraction Methods:**

i.	Bag-of-Words (BoW) with SVM achieved an accuracy of approximately 82.95%, establishing a performance baseline.

ii.	TF-IDF with SVM achieved an accuracy of approximately 86.65%, showcasing the efficacy of term frequency-inverse document frequency weighting.

iii.	N-grams with SVM achieved the highest accuracy among the baseline methods, reaching approximately 85.525%, showcasing the importance of capturing sequential information in emotion analysis.

**2. Intermediate Feature Representation:**

Word2Vec with SVM achieved an accuracy of approximately 44.425%, indicating that word embedding failed to capture the semantic nuances within the dataset.

**3.	State-of-the-art Transformer Model:**

Bidirectional Encoder Representations from Transformers (BERT) achieved an accuracy of approximately 93.275%, outperforming all other models. 

## Comparative Analysis
The traditional methods- BoW, TF-IDF, and N-grams with SVM, provided competitive results, showcasing their relevance and effectiveness for emotion analysis tasks.
The lower accuracy of Word2Vec with SVM suggests that, in this context, the continuous vector representations may not have adequately captured the complexity of emotion-related semantics.
The higher accuracy achieved through BERT highlights the importance of contextualized information for better understanding of nuances in case of emotion analysis tasks.

## Conclusion and Future Work
In conclusion, the results showcase the efficacy of transformer-based models over traditional approaches for emotion analysis. This research contributes valuable insights into selection and understanding of NLP models for emotion analysis tasks.
While this research has provided valuable insights into the comparative performance of various NLP models for emotion analysis, we can further investigate the performance of different transformer-based models and their variants with respect to each other.



