import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def readData():
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv("diabetes.csv", names=col_names)
    return pima

def splitData(X, y):
    return train_test_split(X, y, test_size=0.3)

if __name__ == '__main__':

    # Odczyt danych z pliku
    data = readData()
    data.head()

    # Podział danych na zmienne zależne oraz niezależne
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
    X = data[feature_cols]  # Zmienne niezależne (features)
    y = data.label  # Zmienne zależne (target)

    # Podział danych na część testową oraz uczącą
    X_train, X_test, y_train, y_test = splitData(X, y)

    # Budowanie lasu losowego
    clf = RandomForestClassifier(n_estimators=100)

    # Trenowanie klasyfikatora
    clf.fit(X_train, y_train)

    # Predykcja odpowiedzi dla grupy testowej
    y_pred = clf.predict(X_test)

    print("Precyzja:", metrics.accuracy_score(y_test, y_pred))

    # Tworzenie wyników ważności poszczególnych zmiennych
    feature_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Tworzenie wykresu
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Ważność')
    plt.ylabel('Zmienna niezależna')
    plt.title("Wykres ważności zmiennych niezależnych")
    plt.legend()
    plt.show()


