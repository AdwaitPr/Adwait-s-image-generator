import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris_data = pd.read_csv('/Users/adwaitpratapsingh/Documents/ML.initiate /Iris-model/iris/bezdekIris.data', header=None)

# Assign column names
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Explore the dataset
print(iris_data.head())

# Preprocess the data
X = iris_data.drop('species', axis=1)
y = iris_data['species']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', confusion_matrix(y_test, predictions))
print('Classification Report:\n', classification_report(y_test, predictions))

# Visualize results
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=predictions)
plt.title('Predicted Iris Species')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
