import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('IRIS.csv')

#chanding categorical into numerical form (scaling)
df = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
data['species'] = data['species'].map(df)

#Independent and dependent variables from dataset
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

#training and testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

#Prediction
predicted = dtc.predict([[5,	3.3,	1.4,	0.2]])

print(f"The flower is {'Iris-setosa' if predicted==0 else 'Iris-versicolor' if predicted ==1 else 'Iris-vriginica'}")