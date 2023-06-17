import numpy as np
import pandas as pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as mplt


def main():
    train_data = pandas.read_csv("datasets/credit_train.csv")  # carregar csv de treino
    test_data = pandas.read_csv("datasets/credit_test.csv")  # carregar csv de teste
    train_data.drop(['Months since last delinquent', 'Loan ID', 'Customer ID'], axis=1, inplace=True)
    train_data.dropna(axis=0, inplace=True)  # retirar valores nulos
    train_data.drop_duplicates(inplace=True)  # retirar valores duplicados
    label_enconder = LabelEncoder()
    train_data['Loan Status'] = label_enconder.fit_transform(train_data['Loan Status'])
    train_data['Term'] = label_enconder.fit_transform(train_data['Term'])
    train_data['Purpose'] = label_enconder.fit_transform(train_data['Purpose'])
    train_data['Home Ownership'] = label_enconder.fit_transform(train_data['Home Ownership'])
    train_data['Years in current job'] = label_enconder.fit_transform(train_data['Years in current job'])

    # print(train_data.info())
    y = train_data['Loan Status'].values
    train_data = train_data.drop('Loan Status', axis=1)

    X = train_data.drop(['Years in current job', 'Home Ownership', 'Purpose', 'Monthly Debt', 'Tax Liens', 'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance'], axis=1).values


    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=86)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # pipe = Pipeline([('classifier', RandomForestClassifier())])

    # Create param grid.

    # param_grid = [
    #     {'classifier': [RandomForestClassifier()],
    #      'classifier__n_estimators': list(range(10, 101, 10)),
    #      'classifier__max_features': list(range(6, 32, 5))}
    # ]
    #
    # # Create grid search object
    #
    # clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    #
    # # Fit on data
    #
    # best_clf = clf.fit(x_train, y_train)
    # print(best_clf)

    rand_forest = RandomForestClassifier(max_features=16, max_depth=16, random_state=86)
    rand_forest.fit(x_train, y_train)
    rand_forest.score(x_train, y_train)
    y_pred = rand_forest.predict(x_test)
    print("\nRandom Forest:")
    print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100))
    print("Precision: " + str(precision_score(y_test, y_pred) * 100))
    print("Recall: " + str(recall_score(y_test, y_pred) * 100))
    print("F1-Score: " + str(f1_score(y_test, y_pred) * 100))
    print("AUC: " + str(roc_auc_score(y_test, y_pred) * 100))

    log_reg = LogisticRegression(random_state=86, C=0.0018329807108324356, penalty='l1', solver='liblinear')
    log_reg.fit(x_train, y_train)

    log_reg.score(x_train, y_train)
    y_pred = log_reg.predict(x_test)
    print("\nRegressão Logística:")
    print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100))
    print("Precision: " + str(precision_score(y_test, y_pred) * 100))
    print("Recall: " + str(recall_score(y_test, y_pred) * 100))
    print("F1-Score: " + str(f1_score(y_test, y_pred) * 100))
    print("AUC: " + str(roc_auc_score(y_test, y_pred) * 100))



    # rand_forest = RandomForestClassifier(max_depth=16, max_features=16)
    # rand_forest.fit(x_train, y_train)
    # rand_forest.score(x_train, y_train)
    # y_pred = rand_forest.predict(x_test)
    # print("\nRandom Forest:")
    # print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100))
    # print("Precision: " + str(precision_score(y_test, y_pred) * 100))
    # print("Recall: " + str(recall_score(y_test, y_pred) * 100))
    # print("F1-Score: " + str(f1_score(y_test, y_pred) * 100))
    #
    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # knn.score(x_train, y_train)
    # y_pred = knn.predict(x_test)
    # print("\nKNN:")
    # print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100))
    # print("Precision: " + str(precision_score(y_test, y_pred) * 100))
    # print("Recall: " + str(recall_score(y_test, y_pred) * 100))
    # print("F1-Score: " + str(f1_score(y_test, y_pred) * 100))
    #
    # dec_tree = DecisionTreeClassifier(max_depth=16, max_features=16)
    # dec_tree.fit(x_train, y_train)
    # dec_tree.score(x_train, y_train)
    # y_pred = dec_tree.predict(x_test)
    # print("\nÁrvore de Decisão:")
    # print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100))
    # print("Precision: " + str(precision_score(y_test, y_pred) * 100))
    # print("Recall: " + str(recall_score(y_test, y_pred) * 100))
    # print("F1-Score: " + str(f1_score(y_test, y_pred) * 100))
    #
    # xg = XGBClassifier()
    # xg.fit(x_train, y_train)
    # xg.score(x_train, y_train)
    # y_pred = xg.predict(x_test)
    # print("\nXGB:")
    # print("Accuracy: " + str(accuracy_score(y_test, y_pred) * 100))
    # print("Precision: " + str(precision_score(y_test, y_pred) * 100))
    # print("Recall: " + str(recall_score(y_test, y_pred) * 100))
    # print("F1-Score: " + str(f1_score(y_test, y_pred) * 100))


if __name__ == "__main__":
    main()
