
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
from input_data import NEURAL_NUMBER


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples flag
    plt.contourf(xx, yy, Z, cmap=plt.cm.BuPu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.BuPu,  alpha=0.5)
    # plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.BuPu, alpha=0.8)

def plot_svm(X, y):
    h = .02  # step size in the mesh
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 2.  # SVM regularization parameter

    (perceptron, k_nearest, rbf_svc, poly_svc) = [None] * 4

    # svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    perceptron= MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto', early_stopping=False, learning_rate='adaptive',
       epsilon=1e-06, hidden_layer_sizes=(NEURAL_NUMBER + 1), learning_rate_init=0.01, max_iter=10000, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, warm_start=False).fit(X, y)

    rbf_svc = svm.SVC(kernel='rbf', gamma=1., C=400.).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C*10).fit(X, y)
    k_nearest =  KNeighborsClassifier(3).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['Sklearn Perceptron({0} neurons)'.format(NEURAL_NUMBER),
              'k(3)-nearest neighbors',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((perceptron, k_nearest, rbf_svc, poly_svc)):
        if clf is None : continue
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.BuPu, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.BuPu)
        plt.xlabel('Data axis 1')
        plt.ylabel('Data axis 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.pause(0.1)
