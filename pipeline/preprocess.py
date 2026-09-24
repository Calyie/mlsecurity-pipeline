"""Split a DataFrame into train, validation and test sets and encode the features."""
import re

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def clean_text(column):
    """Lower-case a text column and keep letters, spaces, $ and ! only."""
    return column.fillna("").apply(lambda text: re.sub(r"[^a-z\s$!]", "", text.lower()))


def preprocess_data(df, target, text_cols=None, test_size=0.2, seed=1337):
    """Split `df` and fit an encoder on the training part only.

    Numeric columns are scaled, categorical columns are one-hot encoded and
    text columns become TF-IDF features. Returns
    (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor).
    """
    text_cols = list(text_cols or [])
    X = df.drop(columns=target)
    y = df[target]

    missing = [col for col in text_cols if col not in X.columns]
    if missing:
        raise ValueError(f"Text columns not found: {missing}")

    categorical_cols = [c for c in X.select_dtypes(include=["object", "category", "string"]).columns
                        if c not in text_cols]
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    transformers = [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
    for i, col in enumerate(text_cols):
        text_pipeline = Pipeline([
            ("clean", FunctionTransformer(clean_text)),
            ("tfidf", TfidfVectorizer(stop_words="english", min_df=1, max_df=0.9, ngram_range=(1, 2))),
        ])
        transformers.append((f"text_{i}", text_pipeline, col))
    preprocessor = ColumnTransformer(transformers)

    # test set first, then a validation set carved out of the training part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    X_train = preprocessor.fit_transform(X_train)     # fit on training data only
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    print(f"Train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
