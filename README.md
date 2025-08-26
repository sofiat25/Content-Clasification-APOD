This project aims to build a content classification model for the APOD (Astronomy Picture of the Day) dataset.
The goal is to predict whether a record is:

image

no-image (merging video and other classes)

The workflow uses TF-IDF for text vectorization (title + description), RandomOverSampler for class balancing, and RandomForest as the classifier.

Project Structure
├── notebook.ipynb          # Main notebook with the full workflow
├── data/
│   └── apod.csv            # Input dataset
├── README.md               # This file
└── requirements.txt        # Required Python libraries

Key Libraries

pandas

scikit-learn (TfidfVectorizer, RandomForestClassifier, train_test_split, classification_report)

imblearn (RandomOverSampler)

matplotlib (for plots, e.g., Precision-Recall Curve)

collections.Counter (for class distribution visualization)

Workflow

Text Preprocessing

Concatenate title + description → full_text.

Vectorize using TF-IDF.

Class Balancing

Apply oversampling only on the training set using RandomOverSampler.

Avoid contaminating the test set.

Model Training

Classifier: RandomForestClassifier with class_weight='balanced'.

Train on the balanced data.

Evaluation

Metrics: Precision, Recall, F1-score, Accuracy.

Confusion matrix.

Use predict_proba to adjust the threshold for no-image.

Precision-Recall curve to select the optimal threshold.

Expected Results

The model correctly predicts most image examples.

Adjusting the threshold improves recall for the minority class (no-image).

Final output includes precision, recall, F1-score, confusion matrix, and the Precision-Recall curve.

How to Run

Install dependencies:

pip install -r requirements.txt


Run the notebook.ipynb step by step.

Adjust the threshold for no-image based on the Precision-Recall curve.

Notes

Due to extreme class imbalance, it is recommended to merge minority classes or collect more data.

Metrics like Accuracy can be misleading; use macro F1 and balanced accuracy for fair evaluation.
