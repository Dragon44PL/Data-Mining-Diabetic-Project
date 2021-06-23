import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def readData():
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction',
                 'Age', 'Outcome']
    pima = pd.read_csv("diabetes.csv", names=col_names)
    return pima

def splitData(X, y):
    return train_test_split(X, y, test_size=0.3)

if __name__ == '__main__':

    # Odczyt danych z pliku
    data = readData()
    data.head()

    # Podział danych na zmienne zależne oraz niezależne
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction',
                    'Age']
    target_cols = ['Outcome']
    X = data[feature_cols]  # Zmienne niezależne (features)
    y = data[target_cols]  # Zmienne zależne (target)

    # Podział danych na część testową oraz uczącą
    X_train, X_test, y_train, y_test = splitData(X, y)

    # Budowanie lasu losowego
    clf = RandomForestClassifier(n_estimators=100)

    # Trenowanie klasyfikatora
    clf.fit(X_train, y_train)

    # Predykcja odpowiedzi dla grupy testowej
    y_pred = clf.predict(X_test)

    print("CA:", metrics.accuracy_score(y_test, y_pred))

    print("Precision: ", metrics.precision_score(y_test, y_pred))

    print("F1: ", metrics.f1_score(y_test, y_pred, average='weighted'))

    print("Recall: ", metrics.recall_score(y_test, y_pred, average='weighted'))

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    plt.clf()
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Krzywa ROC dla lasu losowego')
    plt.show()

    # Tworzenie wyników ważności poszczególnych zmiennych
    feature_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Tworzenie wykresu
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Ważność')
    plt.ylabel('Zmienna niezależna')
    plt.title("Wykres ważności zmiennych niezależnych")
    plt.legend()
    plt.show()


