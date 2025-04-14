import numpy as np
import matplotlib.pyplot as plt
from random import randint
from copy import deepcopy

# Modèle
def model(X, W, b) :
    return np.dot(X, W) + b

# Fonction sigmoide (1 + exp(-z)) ^ -1
def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

# Matrice sigmoide
def sigmoid_matrix(Z) :
    return sigmoid(Z)

# Fonction cost
def log_loss(A, Y) :
    return -  np.sum(np.dot(Y.T, np.log(A)) + np.dot((1 - Y).T, np.log(1 - A))) / len(Y)
    
# Gradients
def gradient(X, Y, A) :
    return np.dot(X.T, (A - Y)) / len(Y)

# Descent gradient
def descent_gradient(W, b, X, Y, A, pas_dentrainement) :
    W -= pas_dentrainement * gradient(X, Y, A)
    b -= pas_dentrainement * np.sum(A - Y) / len(Y)
    return W, b

# Predict
def predict(X, W, b) :
    Z = model(X, W, b)
    A = sigmoid_matrix(Z)
    print(A >= 0.5)

# Neuronne
def neuron(X, Y, pas_dentrainement = 0.1 , nombre_iter = 100) :
    W = np.random.randn(2, 1)
    b = randint(1, 10)
    loss = list()

    for i in range(nombre_iter) :
        Z = model(X, W, b)
        A = sigmoid_matrix(Z)
        loss.append(log_loss(A, Y))
        W, b = descent_gradient(W, b, X, Y, A, pas_dentrainement)
    
    # Courbe d'évolution de l'erreur en fonction du nbre d'apprentissage

    # plt.title("Evolution de l'erreur en fonction de l'apprentissage")
    # plt.plot(loss, label = "Fonction coût (Log loss)")
    # plt.xlabel("nombre d'apprentissage")
    # plt.ylabel("Erreur")
    # plt.legend()
    # plt.show()

    return W, b

# Dataset
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
x2 = np.array([8, 5, 9, 7, 10, 6, 10, 11, 8])
X1 = x1
X2 = x2
x1_copy1, x2_copy1 = deepcopy(x1), deepcopy(x2)

x1 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])
x2 = np.array([1, 1.2, 3, 4, 2, 3.4, 1.8, 4, 2.1])
X1 = np.concatenate((X1, x1))
X2 = np.concatenate((X2, x2))
x1_copy2, x2_copy2 = deepcopy(x1), deepcopy(x2)

X = np.hstack((X1.reshape(len(X1), 1), X2.reshape(len(X2), 1)))
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
Y = Y.reshape(len(Y), 1)

W, b = neuron(X, Y)

# Test avec une cellule
X = np.array([[1, 2]])
predict(X, W, b)

# Trace de la frontiere de decision
x_frontiere = np.linspace(min(x1_copy1), max(x1_copy2), 100)
y_frontiere = - (x_frontiere * W[0] + b) / W[1]
plt.title("Neuronne d'itendification de cellule cancereuse")
plt.scatter(x1_copy1, x2_copy1, color = 'orange', label = "Cellule cancereuse")
plt.scatter(x1_copy2, x2_copy2, color = 'red', label = "Cellule saine")
plt.plot(x_frontiere, y_frontiere, color = "blue", label = "Frontière de décision")
plt.legend()
plt.show()


