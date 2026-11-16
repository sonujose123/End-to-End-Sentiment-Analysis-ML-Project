# Amazon Alexa Review Sentiment Analysis

Classifying Amazon Alexa product reviews as **positive** or **negative** using classical NLP preprocessing and a Random Forest classifier.

The project covers the full workflow: exploratory data analysis, text cleaning, bag-of-words feature extraction, model training and evaluation.

## Dataset

`amazon_alexa.tsv` contains **3,150 verified customer reviews** of Amazon Alexa devices.

| Column | Description |
|---|---|
| `rating` | Star rating from 1 to 5 |
| `date` | Date the review was posted |
| `variation` | Product variant (16 in total, e.g. *Charcoal Fabric*, *Walnut Finish*) |
| `verified_reviews` | Review text |
| `feedback` | Sentiment label: `1` = positive, `0` = negative |

Key characteristics:

- **Imbalanced classes:** 2,893 positive (92%) vs. 257 negative (8%) reviews.
- **Ratings skew high:** 2,286 of the reviews are 5 stars.
- **One missing review** in `verified_reviews`, which is dropped.

## Project Workflow

1. **Exploratory data analysis**
   - Null checks, review length, and rating and feedback distributions
   - Mean rating by product variation
   - Word clouds for all reviews, plus words unique to positive and to negative reviews
2. **Text preprocessing**
   - Strip non-alphabetic characters and lowercase
   - Remove English stopwords (NLTK)
   - Stem words with the Porter stemmer
3. **Feature extraction**
   - Bag-of-words with `CountVectorizer` (top 2,500 features)
   - Min-max scaling
4. **Modelling**
   - 70/30 train/test split (`random_state=15`)
   - `RandomForestClassifier` trained on the scaled features
5. **Evaluation**
   - Training and test accuracy
   - Confusion matrix

## Tech Stack

Python · pandas · NumPy · scikit-learn · NLTK · Matplotlib · Seaborn · WordCloud · XGBoost

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost wordcloud
```

### Run

Make sure `amazon_alexa.tsv` is in the same folder as the script, then:

```bash
python sentiment_analysis.py
```

> **Note:** The script was exported from a Google Colab notebook and contains the notebook command `!pip install wordcloud`. This line is not valid Python outside a notebook, so remove it (the package is installed above) or run the code in Google Colab / Jupyter instead.

## Repository Structure

```
.
├── amazon_alexa.tsv         # Dataset
├── sentiment_analysis.py    # EDA, preprocessing, training, evaluation
└── README.md
```

## Results

Running the script prints training accuracy, test accuracy and a confusion matrix for the Random Forest model.

Because roughly 92% of reviews are positive, accuracy alone can look strong even for a model that favours the majority class. Check the confusion matrix, and look at precision and recall for the negative class, when judging performance.

## Possible Improvements

- Handle class imbalance (class weights, SMOTE, or resampling)
- Report precision, recall and F1, not just accuracy
- Tune hyperparameters with `GridSearchCV` and `StratifiedKFold`
- Compare against other models such as XGBoost and Decision Tree
- Try TF-IDF features or transformer-based embeddings
- Save the fitted vectorizer, scaler and model with `pickle` for reuse

## License

Add a license of your choice (e.g. MIT) before publishing.

## Author

**Sonu Jose**
