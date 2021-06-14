import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from matplotlib import pyplot as plt

def readData():
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction',
                 'Age', 'Outcome']
    data = pd.read_csv("diabetes.csv", names=col_names)
    return data

if __name__ == '__main__':

    # Odczyt danych z pliku
    data = readData()
    data.head()

    # Podział danych na zmienne zależne oraz niezależne
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction',
                    'Age']
    target_cols = ['Outcome']
    X = data[feature_cols]  # Zmienne niezależne (features)
    y = data.Outcome # Zmienne zależne (target)

    # Podział danych na część testową oraz uczącą
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Budowanie drzewa decyzyjnego
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Trenowanie klasyfikatora
    clf = clf.fit(X_train, y_train)

    # Predykcja odpowiedzi dla grupy testowej
    y_pred = clf.predict(X_test)

    print("Precyzja:", metrics.accuracy_score(y_test, y_pred))

    # Utworzenie wykresu dla drzewa decyzyjnego
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       feature_names=feature_cols,
                       class_names=['not-diabetic', 'diabetic'],
                       filled=True)

    fig.savefig("decision_tree.png")
