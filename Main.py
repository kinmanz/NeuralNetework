
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from graph import plot_decision_boundary, plot_svm
from input_data import NOISE
from input_data import NUMBER_OF_DATA, NEURAL_NUMBER
from csv_myreader import get_csv_data

np.random.seed(0)

#
# NEURAL_NUMBER = 50
# NUMBER_OF_DATA = 2000
# X, y = get_csv_data(NUMBER_OF_DATA)
# epsilon = 0.0002  # learning rate for gradient descent
# reg_lambda = 0.0002  # regularization strength
# epsilon_moving= 10**-8  # regularization strength


X, y = make_moons(NUMBER_OF_DATA, noise=NOISE)
# Gradient descent parameters
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01  # regularization strength
epsilon_moving= 10**-6  # regularization strength
# X - two dimensional points
# y - point cluster 0 or 1

print(X, y)
fig = plt.figure(1)
plt.title("Data", fontsize = 20)

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.BuPu, alpha=0.8)
# plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.BuPu, alpha=0.8)
# plt.pause(1); plt.close()
plt.show()

# fig2 = plt.figure(2)
# plot_svm(X, y)

plt.figure(1)
plt.title("Percepteron with " + str(NEURAL_NUMBER) + " neurons")

# Male and Female weight and height
num_examples = len(X)  # training set size
network_input_dim = 2  # input layer dimensionality
network_output_dim = 2  # output layer dimensionality



# should define male of female for by wight and height

# First attempt to use Perceptron
# The input to the network will be x- and y- coordinates and its output will be two probabilities,
# one for class 0 Female and one for class 1 Male.
# In our class data.


# evaluate loss
def calculate_loss(model):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    # apply softmax
    exp_scores = np.exp(z2)
    # get all probabilities from formula (two exactly)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # It's network output!

    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
# and returns the class with the highest probability.
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    global epsilon
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(network_input_dim, nn_hdim) / np.sqrt(network_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, network_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, network_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1


        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        derev_norm = np.linalg.norm(epsilon * dW1) + np.linalg.norm(epsilon * db1) + np.linalg.norm(epsilon * dW2) + np.linalg.norm(epsilon * db2)

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if derev_norm < epsilon_moving:
            print("Loss iteration :" , i)
            print("Stop ! Iteration:" , i, " Epsilon stop: ", epsilon_moving)
            plot_decision_boundary(lambda x: predict(model, x), X, y)
            # plt.pause(10)
            break

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            epsilon /= (1.3**(i/1000))
            print("Error function (Loss) after iteration %i: %f" % (i, calculate_loss(model)))
            plot_decision_boundary(lambda x: predict(model, x), X, y)
            plt.pause(0.01)
            # plt.show()
            plt.close(fig)


    return model


# BUILD
model = build_model(NEURAL_NUMBER, print_loss=True)

plt.pause(1000)


