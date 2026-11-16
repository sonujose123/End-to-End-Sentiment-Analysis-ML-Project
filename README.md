# Amazon Alexa Review Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19RU6pSilm5sFOhFYH5nBzAkNtCB1CeIR)

Classifying Amazon Alexa product reviews as **positive** or **negative** using classical NLP preprocessing and a Random Forest classifier, built in a Jupyter notebook on Google Colab.

The notebook covers the full workflow: exploratory data analysis, text cleaning, bag-of-words feature extraction, model training and evaluation.

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

## Notebook Workflow

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

Python · Jupyter / Google Colab · pandas · NumPy · scikit-learn · NLTK · Matplotlib · Seaborn · WordCloud · XGBoost

## How to Run

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page.
2. Upload `amazon_alexa.tsv` to the Colab session: open the **Files** panel on the left and use **Upload**, or run:
   ```python
   from google.colab import files
   files.upload()
   ```
3. Run all cells (**Runtime → Run all**). The first cells install `wordcloud` and download the NLTK stopwords automatically.

### Option 2: Local Jupyter Notebook

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost wordcloud notebook
jupyter notebook
```

Open `Sentiment Analysis.ipynb`, make sure `amazon_alexa.tsv` is in the same folder, and run all cells.

## Repository Structure

```
.
├── Sentiment Analysis.ipynb   # Main notebook (EDA, preprocessing, training, evaluation)
├── sentiment_analysis.py      # Script export of the notebook (for reference)
├── amazon_alexa.tsv           # Dataset
└── README.md
```

> `sentiment_analysis.py` is an auto-generated export of the Colab notebook. It still contains notebook-only commands such as `!pip install wordcloud`, so run the `.ipynb` rather than the `.py` file.

## Results

The notebook prints training accuracy, test accuracy and a confusion matrix for the Random Forest model.

Because roughly 92% of reviews are positive, accuracy alone can look strong even for a model that favours the majority class. Check the confusion matrix, and look at precision and recall for the negative class, when judging performance.

## Possible Improvements

- Handle class imbalance (class weights, SMOTE, or resampling)
- Report precision, recall and F1, not just accuracy
- Tune hyperparameters with `GridSearchCV` and `StratifiedKFold`
- Compare against other models such as XGBoost and Decision Tree
- Try TF-IDF features or transformer-based embeddings
- Save the fitted vectorizer, scaler and model with `pickle` for reuse


## Author

**Sonu Jose**
