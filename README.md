# Dry Bean Classification

## What this project does

For this assignment, I used the Dry Bean dataset to compare five classification algorithms. The task is to predict the bean variety using the numerical measurements in the dataset.

## Dataset

- Source: UCI Machine Learning Repository – Dry Bean Dataset
- Records: 13,611
- Input features: 16 numerical features
- Target: `Class`
- Number of classes: 7
- Dataset page: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The original Excel file is kept in the `data` folder. The dataset has 68 duplicate rows and no missing values.

## Models used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

StandardScaler is included in the pipelines for Logistic Regression, KNN and SVM. The tree and Gaussian Naive Bayes models are trained without scaling.

## How I split the data

I used an 80:20 train-test split with `random_state=42` and stratified sampling. This keeps the class proportions similar in the training and test sets. The split contains 10,888 training records and 2,723 test records.

## Results

The following values were calculated by running the models on the held-out test set. AUC is calculated using one-vs-rest macro averaging. Precision, recall and F1 are macro averages.

| Model               |   Accuracy |   AUC (OvR macro) |   Precision (macro) |   Recall (macro) |   F1 (macro) |    MCC |
|:--------------------|-----------:|------------------:|--------------------:|-----------------:|-------------:|-------:|
| SVM                 |     0.9221 |            0.9949 |              0.9359 |           0.933  |       0.9343 | 0.9059 |
| Logistic Regression |     0.9214 |            0.9948 |              0.9354 |           0.9321 |       0.9335 | 0.905  |
| KNN                 |     0.9166 |            0.9833 |              0.932  |           0.9271 |       0.9293 | 0.8992 |
| Decision Tree       |     0.8917 |            0.9448 |              0.9084 |           0.9088 |       0.9085 | 0.8691 |
| Naive Bayes         |     0.7639 |            0.9672 |              0.7744 |           0.7694 |       0.7677 | 0.7154 |

## Observations

- SVM gave the highest macro F1 score in this run, with Logistic Regression very close behind.
- KNN also performed well, while the Decision Tree score was lower than the top three models.
- Gaussian Naive Bayes had the lowest accuracy and macro F1 among these five models.
- The AUC scores are high across the models, although the classification metrics show that their prediction quality is not identical.
- Based on this split, SVM is the strongest model overall by macro F1. The difference between SVM and Logistic Regression is small.

These are results from one train-test split, so they should be interpreted as a comparison for this experiment rather than a guarantee of performance on every new dataset.

## Run locally

Install the packages:

```bash
pip install -r requirements.txt
```

Train and save the models:

```bash
python src/train_models.py
```

Start the Streamlit app:

```bash
streamlit run app.py
```

The app loads the saved model files. It does not train models when a user uploads a CSV.

## Files

- `app.py` – Streamlit application
- `src/train_models.py` – training and evaluation code
- `data/Dry_Bean_Dataset.xlsx` – original dataset
- `model/` – five saved models and model metadata
- `test_data.csv` – held-out test data with the target column
- `model_comparison.csv` – model comparison results
- `requirements.txt` – Python dependencies

## Links

- GitHub repository: [Add your GitHub repository URL here]
- Streamlit app: [Add your deployed Streamlit URL here]

## Note

Before submission, I will upload this project to my GitHub repository and deploy the app on Streamlit Community Cloud. I will replace the two link placeholders above with the actual links.
