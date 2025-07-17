# 🌍 Language Identification using Machine Learning and Deep Learning

This project focuses on detecting the language of a given text using two approaches:

* A traditional **Machine Learning** pipeline with `CountVectorizer` and `MultinomialNB`.
* A **Deep Learning** model using `Bidirectional LSTM` and an **Attention** mechanism.

The notebook demonstrates the **entire workflow**: data cleaning, preprocessing, modeling, training, and evaluation.

---

## 📂 Dataset

* Source: [Language Identification Dataset on Kaggle](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)
* The dataset contains thousands of text samples from multiple languages.

---

## 🧹 Preprocessing

The text data was cleaned using basic regular expression techniques to improve data quality:

* Removed punctuations
* Removed excessive whitespaces
* Removed empty or invalid records

```python
def clean_text(text):
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    text_clean = re.sub(r'\s+', ' ', text_no_punct)
    return text_clean.strip()
```

---

## 🧠 Models

### 🔹 1. Classical Machine Learning Model

* **Vectorizer**: `CountVectorizer`
* **Classifier**: `Multinomial Naive Bayes`
* **Train/Test Split**: Stratified 80/20
* **Accuracy**: \~92%

### 🔹 2. Deep Learning Model

* **Embedding Layer**
* **Bidirectional LSTM**
* **Attention Layer**
* **Dropout + Dense Layers**
* **Final Accuracy**: \~97.33%

```python
inputs = Input(shape=(maxlen,))
x = Embedding(input_dim=len(word_idx) + 1, output_dim=128, input_length=maxlen)(inputs)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Attention()([x, x])
x = GlobalAveragePooling1D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(class_nums, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

---

## 📊 Results

| Model              | Accuracy |
| ------------------ | -------- |
| Multinomial NB     | 92%      |
| BiLSTM + Attention | 97.33%   |

---

## 📚 Tech Stack

* Python 🐍
* Scikit-learn
* TensorFlow / Keras
* NumPy / Pandas / Matplotlib
* Kaggle Datasets

---

## 🌟 Key Takeaways

* Classical ML can still provide strong baselines.
* Deep learning shows clear improvements in performance for sequence-based problems.
* Clean and well-preprocessed text significantly improves model quality.
* Demonstrates how to combine NLP preprocessing with different model architectures.

---

## ✅ Author

> Mohammad (a.k.a. **JafaDsX**) – Aspiring NLP Data Scientist
> 💼 [LinkedIn](#) | 📩 [jafadsx@gmail.com](mailto:jafadsx@gmail.com) | 🧠 Always Learning...

---

## 📍 Project Purpose

This project was built as a portfolio piece to demonstrate:

* Practical NLP skills
* Comparative modeling (ML vs DL)
* Clean, documented code in Jupyter Notebook format
* Real-world dataset handling

If you liked this project or found it useful, feel free to connect or leave a star on GitHub ⭐
