import gensim
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

medical_corpus=["Diabetes is a chronic disease that affects the way the body processes blood sugar.",
                "Hypertension, or high blood pressure, can lead to heart disease and stroke.",
                "The patient was diagnosed with pneumonia and prescribed antibiotics.",
                "Insulin therapy is commonly used for type 1 diabetes patients.",
                "Cardiovascular diseases are the leading causes of death worldwide."
                ]

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words= set(stopwords.words("english"))
processed_corpus=[]

for sentence in medical_corpus:
  tokens=word_tokenize(sentence.lower())
  tokens=[word for word in tokens if word not in stop_words and word not in string.punctuation]
  processed_corpus.append(tokens)

print("processed corpus : ",processed_corpus)
word2vec_model= Word2Vec(sentences=processed_corpus, vector_size=300,window=5,min_count=1,workers=4)

word2vec_model.save("medical_word2vec.model")
word2vec_model= Word2Vec.load("medical_word2vec.model")
print("Vocabulary:", list(word2vec_model.wv.index_to_key))
print("similar words to 'diabetes' : ",word2vec_model.wv.most_similar("diabetes"))
result=word2vec_model.wv.most_similar(positive=['hypertension','heart'],negative=['stroke'])

print("Word vector arithmetic result: ", result)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
words=list(word2vec_model.wv.index_to_key)
word_vectors= [ word2vec_model.wv[word] for word in words]
pca= PCA(n_components=2)
reduced_vectors=pca.fit_transform(word_vectors)

plt.figure(figsize=(8,6))
for i, word in enumerate(words):
  plt.scatter(reduced_vectors[i,0], reduced_vectors[i,1])
  plt.annotate(word,(reduced_vectors[i,0],reduced_vectors[i,1]))
plt.title("word embedding visualization(PCA)")
plt.show()