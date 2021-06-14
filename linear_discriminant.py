import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


def readData():
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction',
                 'Age', 'Outcome']
    data = pd.read_csv("diabetes.csv", names=col_names)
    return data

if __name__ == '__main__':
    data = readData()
    target_names = ['Outcome']
    X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction',
                    'Age']]
    y = data[target_names]

    # Podział danych na część testową oraz uczącą
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Budowanie modelu dyskryminacyjnego
    lda = LinearDiscriminantAnalysis(n_components=1)

    # Trenowanie klasyfikatora
    X_r2 = lda.fit(X_train, y_train).transform(X_train)

    # Predykcja odpowiedzi dla grupy testowej
    lda = lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    print("Precyzja:", metrics.accuracy_score(y_test, y_pred))

    plt.figure()
    colors = ['navy', 'turquoise']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], ['Outcome']):
        plt.scatter(X_r2[y_train == i], X_r2[y_train == i], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Wykres modelu dyskryminacyjnego dla zbioru danych')

    plt.show()
