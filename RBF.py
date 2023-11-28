from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 784)).astype('float32') / 255
x_test = x_test.reshape((10000, 784)).astype('float32') / 255
 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10) 

def build_model(hidden_layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(784,)))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


models = []
for i, (layers, neurons) in enumerate([(1, 32), (2, 64), (3, 128)]):
    model = build_model(hidden_layers=layers, neurons=neurons)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Training model {i+1} with {layers} hidden layers and {neurons} neurons each.")
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
    models.append((model, history))


for i, (model, history) in enumerate(models):
    test_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Model {i+1} Accuracy: {test_accuracy:.4f}")




def rbf_features(X, centers, gamma):
    d = -gamma * np.linalg.norm(X[:, np.newaxis, :] - centers, axis=2) ** 2
    return np.exp(d)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(x_train)


rbf_model = make_pipeline(
    FunctionTransformer(rbf_features, kw_args={'centers': kmeans.cluster_centers_, 'gamma': 1.0}),
    Ridge(alpha=1.0)
)


rbf_model.fit(x_train, y_train)

y_pred = rbf_model.predict(x_test)

y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)


conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print('Confusion Matrix:')
print(conf_matrix)


accuracy = accuracy_score(y_true_labels, y_pred_labels)
print('Accuracy:', accuracy)


class_report = classification_report(y_true_labels, y_pred_labels)
print('Classification Report:')
print(class_report)
