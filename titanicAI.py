import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)

previsoes = model.predict(X_test)

accuracy = accuracy_score(y_test, previsoes)
print(f"Acurácia do Modelo: {accuracy*100:.2f}%")

sns.heatmap(confusion_matrix(y_test, previsoes), annot=True, fmt='d', cmap='Reds')
plt.title('Matriz de Confusão: Titanic')
plt.ylabel('Realidade (0=Morreu, 1=Viveu)')
plt.xlabel('Previsão da IA')
plt.show()