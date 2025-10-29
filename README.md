# Amazon Review Sentiment Analysis

This repository contains a machine learning project focused on sentiment analysis of the large-scale Amazon Reviews dataset. The primary goal is to preprocess millions of text reviews and build classification models to accurately determine whether a review is positive or negative. This project demonstrates a complete NLP pipeline, from raw text cleaning to model evaluation, achieving approximately 88% accuracy.

![sentiment analysis](https://www.revuze.it/blog/wp-content/uploads/sites/2/2020/03/Amazon-Review-Analysis.png)


## Project Overview

The project follows a structured approach to tackle the sentiment analysis problem:

1.  **Data Loading**: The large dataset, compressed in `.bz2` files, is loaded into a pandas DataFrame.
2.  **Text Preprocessing**: The raw review text is thoroughly cleaned by:
    *   Converting text to lowercase.
    *   Removing common English stopwords.
    *   Applying stemming to reduce words to their root form.
    *   Stripping out URLs, special characters, and other noise.
3.  **Feature Extraction**: The cleaned text is converted into a numerical format using `HashingVectorizer`, which is efficient for large datasets.
4.  **Model Training**: Two different classification models are trained on the vectorized data:
    *   **Logistic Regression**
    *   **Linear Support Vector Classifier (LinearSVC)**
5.  **Evaluation**: The performance of each model is measured using accuracy and a detailed classification report (precision, recall, F1-score).

## Dataset

The project utilizes the **Amazon Reviews Dataset**, which consists of 4 million reviews collected from Amazon. The dataset is balanced, with 2 million positive reviews (label `2`) and 2 million negative reviews (label `1`).

## Results

Both models performed well on the test set of 400,000 reviews:

*   **Logistic Regression**: Accuracy of **88.05%**
*   **Linear SVC**: Accuracy of **88.00%**

The results demonstrate that even with a memory-efficient vectorizer like `HashingVectorizer`, it is possible to achieve high accuracy on a large-scale text classification task.

## How to Run This Project

To run this project on your own machine, follow these steps:

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/MohamedAchraf22/Amazon-Reviews-Sentiment-Analysis.git
    cd Amazon-Reviews-Sentiment-Analysis
    ```

2.  **Download the Dataset**
    Download the Amazon Reviews dataset from a source like Kaggle. You will need the `train.ft.txt.bz2` and `test.ft.txt.bz2` files.

3.  **Update File Paths**
    In the Jupyter Notebook, update the `paths` variable in the second code cell to point to the location of the dataset files on your machine.

4.  **Run the Notebook**
    Launch Jupyter Notebook or Jupyter Lab and open the `.ipynb` file to execute the cells.
    ```sh
    jupyter notebook "Sentiment_Analysis_Documented.ipynb"
    ```

## Technologies Used

*   **Python 3**
*   **Pandas**: For data manipulation and analysis.
*   **NLTK**: For natural language processing tasks like stopword removal and stemming.
*   **Scikit-learn**: For feature extraction, model training, and evaluation.
*   **Jupyter Notebook**: For interactive development and documentation.

## Future Improvements

*   **Use `TfidfVectorizer`**: Instead of `HashingVectorizer`, using `TfidfVectorizer` could improve accuracy by giving more weight to important words.
*   **Hyperparameter Tuning**: Perform a systematic search using `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the models.
*   **Explore Deep Learning Models**: Implement more advanced models like LSTMs, GRUs, or Transformers (e.g., BERT) for potentially higher accuracy.
*   **Use `TfidfVectorizer`**: Instead of `HashingVectorizer`, using `TfidfVectorizer` could improve accuracy by giving more weight to important words.
*   **Hyperparameter Tuning**: Perform a systematic search using `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the models.
*   **Explore Deep Learning Models**: Implement more advanced models like LSTMs, GRUs, or Transformers (e.g., BERT) for potentially higher accuracy.
