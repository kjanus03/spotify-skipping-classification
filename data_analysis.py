import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def analyze_data_with_dtc(X_train, X_test, y_train) -> list[int]:
    print(pd.Series(y_train).value_counts())

    # C = 2.0
    # model = LinearSVC(C=C)
    # model = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    model = DecisionTreeClassifier(class_weight={0: 5, 1: 1}, random_state=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
