import pandas as pd
import numpy as np

from scipy.stats import entropy

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import mutual_info_score

import warnings
warnings.filterwarnings("ignore")


class Discriminator:

    def __init__(self, original_df: pd.DataFrame, generated_df: pd.DataFrame):
        self.original_df = original_df
        self.generated_df = generated_df

    def score(self):
        cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
        con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

        # ohe
        df1 = self.original_df
        df1 = pd.get_dummies(df1, columns=cat_cols, drop_first=True)
        original_X = df1.drop(['output'], axis=1)
        original_y = df1[['output']]
        scaler = RobustScaler()
        original_X[con_cols] = scaler.fit_transform(original_X[con_cols])

        df2 = self.generated_df
        df2[cat_cols] = df2[cat_cols].astype(int)
        df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=True)
        df1_not_df2 = df1.columns.difference(df2.columns)
        df2[df1_not_df2] = 0
        generated_X = df2.drop(['output'], axis=1)
        generated_y = df2[['output']]
        scaler = RobustScaler()
        generated_X[con_cols] = scaler.fit_transform(generated_X[con_cols])

        # train/test split
        original_X_train, original_X_test, original_y_train, original_y_test = \
            train_test_split(original_X, original_y, test_size=0.2, random_state=42)
        generated_X_train, generated_X_test, generated_y_train, generated_y_test = \
            train_test_split(generated_X, generated_y, test_size=0.2, random_state=42)

        # таблица с результатами
        results = []
        columns = ['Модель', 'Качество на исходном датасете', 'Качество на сгенерированном датасете']

        # SVM
        original_clf = SVC(kernel='linear', C=1, random_state=42).fit(original_X_train, original_y_train)
        y_pred = original_clf.predict(original_X_test)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_clf = SVC(kernel='linear', C=1, random_state=42).fit(generated_X_train, generated_y_train)
        y_pred = generated_clf.predict(original_X_test)
        generated_accuracy = f1_score(original_y_test, y_pred)
        results.append(['SVM', original_accuracy, generated_accuracy])

        # Logistic regression
        original_logreg = LogisticRegression()
        original_logreg.fit(original_X_train, original_y_train)
        y_pred_proba = original_logreg.predict_proba(original_X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_logreg = LogisticRegression()
        generated_logreg.fit(generated_X_train, generated_y_train)
        y_pred_proba = generated_logreg.predict_proba(original_X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        generated_accuracy = f1_score(original_y_test, y_pred)
        results.append(['Логистическая регрессия', original_accuracy, generated_accuracy])

        # Decision Tree
        original_dt = DecisionTreeClassifier(random_state=42)
        original_dt.fit(original_X_train, original_y_train)
        y_pred = original_dt.predict(original_X_test)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_dt = DecisionTreeClassifier(random_state=42)
        generated_dt.fit(generated_X_train, generated_y_train)
        y_pred = generated_dt.predict(original_X_test)
        generated_accuracy = f1_score(original_y_test, y_pred)
        results.append(['Decision Tree', original_accuracy, generated_accuracy])

        # Random Forest
        original_rf = RandomForestClassifier()
        original_rf.fit(original_X_train, original_y_train)
        y_pred = original_rf.predict(original_X_test)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_rf = RandomForestClassifier()
        generated_rf.fit(generated_X_train, generated_y_train)
        y_pred = generated_rf.predict(original_X_test)
        generated_accuracy = f1_score(original_y_test, y_pred)
        results.append(['Random Forest', original_accuracy, generated_accuracy])

        # Gradient Boosting
        original_gbt = GradientBoostingClassifier(n_estimators=300,
                                                  max_depth=1,
                                                  subsample=0.8,
                                                  max_features=0.2,
                                                  random_state=42)

        original_gbt.fit(original_X_train, original_y_train)
        y_pred = original_gbt.predict(original_X_test)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_gbt = GradientBoostingClassifier(n_estimators=300,
                                                   max_depth=1,
                                                   subsample=0.8,
                                                   max_features=0.2,
                                                   random_state=42)

        generated_gbt.fit(generated_X_train, generated_y_train)
        y_pred = generated_gbt.predict(original_X_test)
        generated_accuracy = f1_score(original_y_test, y_pred)
        results.append(['Gradient Boosting Classifier', original_accuracy, generated_accuracy])

        # Baseline models
        original_dummy_clf = DummyClassifier(strategy="uniform", random_state=42)
        original_dummy_clf.fit(original_X_train, original_y_train)
        y_pred = original_dummy_clf.predict(original_X_test)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_dummy_clf = DummyClassifier(strategy="uniform", random_state=42)
        generated_dummy_clf.fit(generated_X_train, generated_y_train)
        y_pred = generated_dummy_clf.predict(original_X_test)
        generated_accuracy = f1_score(original_y_test, y_pred)

        results.append(['Uniform Dummy Classifier', original_accuracy, generated_accuracy])

        original_dummy_clf = DummyClassifier(strategy="most_frequent")
        original_dummy_clf.fit(original_X_train, original_y_train)
        y_pred = original_dummy_clf.predict(original_X_test)
        original_accuracy = f1_score(original_y_test, y_pred)

        generated_dummy_clf = DummyClassifier(strategy="most_frequent")
        generated_dummy_clf.fit(generated_X_train, generated_y_train)
        y_pred = generated_dummy_clf.predict(original_X_test)
        generated_accuracy = f1_score(original_y_test, y_pred)

        results.append(['Most Frequent Dummy Classifier', original_accuracy, generated_accuracy])

        # таблица с результатами
        table = pd.DataFrame(data=results, columns=columns)
        return table.to_string()

    def mi(self):
        x = self.original_df.values.flatten()
        y = self.generated_df.values.flatten()
        return mutual_info_score(x, y)

    def entropy(self):
        x_series = pd.Series(self.original_df.values.flatten())
        y_series = pd.Series(self.generated_df.values.flatten())
        return min(entropy(x_series.value_counts()), entropy(y_series.value_counts()))

