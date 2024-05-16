from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import pandas as pd
from ydata_profiling import ProfileReport


def print_results(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))


def create_report(df: pd.DataFrame, output_directory: str) -> None:
    profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
    report_file = 'streamingz _history_report.html'
    profile.to_file(output_directory + report_file)

    summary_df = df.describe()
    summary_df.loc['missing'] = df.isnull().sum()
    summary_df.loc['distinct_count'] = df.nunique()

    latex_table = summary_df.to_latex()
    with open(output_directory + 'summary_table.tex', 'w') as f:
        f.write(latex_table)

