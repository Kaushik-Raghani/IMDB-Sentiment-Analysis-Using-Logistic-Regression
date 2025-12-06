
# IMDB Sentiment Analysis Using Logistic Regression




## 🧠 Overview
This project demonstrates a complete machine learning pipeline for sentiment analysis on movie reviews from the IMDB dataset. It includes data preprocessing, feature extraction using TF-IDF, model training with logistic regression, and performance evaluation using standard metrics. The goal is to classify reviews as either positive or negative based on their textual content.
## ❓ Problem Statement
Online reviews play a crucial role in shaping consumer decisions. However, manually analyzing thousands of reviews is impractical. This project aims to automate sentiment classification of IMDB movie reviews to determine whether a review expresses a positive or negative sentiment using natural language processing and machine learning techniques.
## 📂 Dataset
Source: IMDB Dataset of 50,000 movie reviews

Columns:

review: Textual content of the movie review

sentiment: Label indicating sentiment (positive or negative)
## 🛠️ Tools & Technologies
Programming Language: Python

Libraries:

pandas, numpy: Data manipulation

nltk: Text preprocessing (tokenization, stopword removal, stemming, lemmatization)

sklearn: Machine learning (TF-IDF, logistic regression, metrics)

matplotlib, seaborn: Data visualization

wordcloud: Visualizing frequent terms

Model: Logistic Regression

Vectorization: TF-IDF (TfidfVectorizer)

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
## 🔍 Key insights
Balanced Dataset: Nearly equal distribution of positive and negative reviews ensures unbiased model training.

Text Cleaning: Lowercasing, punctuation removal, stopword filtering, and chat abbreviation expansion improved data quality.

Feature Engineering: TF-IDF captured term importance effectively for classification.

Model Performance:

Accuracy: ~89.2%

Precision/Recall/F1: Balanced across both classes

Confusion Matrix: Shows strong predictive power with minimal misclassification
## 📊 Visualizations
Word Cloud: Highlights most frequent words in reviews post-cleaning.

Confusion Matrix: Visual breakdown of true vs. predicted labels.
## Conclusion
This project successfully demonstrates how to build a robust sentiment analysis model using logistic regression and TF-IDF features. The pipeline is modular, reproducible, and well-suited for extension to other text classification tasks. With ~89% accuracy, the model performs reliably and can be deployed for real-world review analysis.
## Contributing
Contributions to this project are welcome! If you have ideas for improvements or additional insights, please open an issue or a pull request. Your contributions will be greatly appreciated.
## 📬 Contact Information
LinkedIn: kaushik-raghani

Email: kaushikraghani23@gmail.com