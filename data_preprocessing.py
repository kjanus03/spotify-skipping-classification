import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def process_platform(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df['platform'].str.lower().str.contains('windows', na=False),
        df['platform'].str.lower().str.contains('android', na=False),
        df['platform'].str.lower().str.contains('ios', na=False)
    ]
    choices = ['PC', 'Android', 'iPhone']
    df['platform'] = np.select(conditions, choices, default='Other').astype(str)
    return df


def process_skipped(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['skipped'], inplace=True)
    df['skipped'] = df['skipped'].apply(lambda x: 1 if x else 0)
    return df


def process_shuffle(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['shuffle'], inplace=True)
    df['shuffle'] = df['shuffle'].apply(lambda x: 1 if x else 0)
    return df


def preprocess_data(df: pd.DataFrame, categorical_features: list[str], numerical_features: list[str]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df.dropna(subset=categorical_features + numerical_features, inplace=True)

    transformer = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), (
        'cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', transformer)])

    X = df[categorical_features + numerical_features]
    y = df['skipped']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_preprocessed = pipeline.fit_transform(X_train)
    X_test_preprocessed = pipeline.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test


def undersample_preprocessed_data(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    indices_class_0 = np.where(y_train == 0)[0]
    indices_class_1 = np.where(y_train == 1)[0]

    np.random.seed(42)

    # losowo wybieramy tyle wartosci z klasy 1, ile jest wartosci z klasy 0
    undersampled_indices_class_1 = np.random.choice(indices_class_1, size=len(indices_class_0), replace=False)
    undersampled_indices = np.concatenate([indices_class_0, undersampled_indices_class_1])

    return X_train[undersampled_indices], y_train.iloc[undersampled_indices]


def oversample_preprocessed_data(X_train: sparse.csr_matrix, y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    indices_class_0 = np.where(y_train == 0)[0]
    indices_class_1 = np.where(y_train == 1)[0]

    np.random.seed(42)

    # losowo wybieramy tyle wartosci z klasy 0, ile jest wartosci z klasy 1
    oversampled_indices_class_0 = np.random.choice(indices_class_0, size=len(indices_class_1), replace=True)
    oversampled_indices = np.concatenate([indices_class_1, oversampled_indices_class_0])

    return X_train[oversampled_indices], y_train.iloc[oversampled_indices]


def remove_outliers(df: pd.DataFrame, numerical_features: list[str], lower_quantile: float = 0.005, upper_quantile: float = 0.995) -> pd.DataFrame:
    for feature in numerical_features:
        lower_bound = df[feature].quantile(lower_quantile)
        upper_bound = df[feature].quantile(upper_quantile)
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

