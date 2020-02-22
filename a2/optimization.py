# optimization.py

import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

alpha_map = {
    "NN" : (0.0112, 600)
}
def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--tip_point', type=float, default=0.0, help='find tipping point starting from provided value.')
    parser.add_argument('--best_step', default=False, help='finding the best step', action='store_true')
    args = parser.parse_args()
    return args


def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2


def quadratic_grad(x1, x2):
    """
    Should return a numpy array containing the gradient of the quadratic function defined above evaluated at the point
    :param x1: first coordinate
    :param x2: second coordinate
    :return: a two-dimensional numpy array containing the gradient
    """
    return np.array([2 * x1 - 2, 16 * x2 -16])

def dist_from_optimum(x, y):
    return y**2 + (1-x[0]) ** 2 + (1-x[1]) ** 2

def sgd_test_quadratic(args, empir=False):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    # Track the points visited here
    points_history = []
    curr_point = np.array([0, 0])
    for iter in range(0, args.epochs):
        next_point = curr_point - args.lr * quadratic_grad(curr_point[0], curr_point[1])
        points_history.append(curr_point)
        print("Point after epoch %i: %s" % (iter, repr(next_point)))
        curr_point = next_point
        
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()

def tipping_point(args):
    start = args.tip_point
    for i in range(1,51):
        args.lr = start * i
        sgd_test_quadratic(args, True)

def best_step_size(args):
    start = 0.11
    steps = []
    iters = []
    while start >= 0:
        args.lr = start
        steps.append(start)
        iters.append(sgd_test_quadratic(args, True))
        start -= 0.01
    plt.plot(steps, iters)
    plt.show()

def ReLU(x):
    return x * (x > 0)

def ReLU_deriv(x):
    return (x > 0) * 1

def forward(x, W1, W2):
    """
    Forward computation of the neural network defined by parameters W1 and W2: softmax(W2 ReLU(W1 x))
    :param x: input point (2-dimensional)
    :param W1: 2 x hidden size
    :param W2: hidden size x num classes
    :return: hidden layer activations (after nonlinearity) and output probabilities
    """
    hidden = np.matmul(W1, x)
    hidden = ReLU(hidden)
    final = np.matmul(W2, hidden)
    final_probs = scipy.special.softmax(final)
    return hidden, final_probs


def backward(x, y, hidden, final_probs, W2):
    """
    Backward computation
    :param x: input point
    :param y: label
    :param hidden: hidden layer activations (after nonlinearity), from forward
    :param final_probs: output probabilities, from forward
    :param W2: weight matrix
    :return: gradients for W1 and W2
    """
    gold_signal = (np.array([1., 0.]) if y == 0 else np.array([0., 1.]))
    err_signal_output = gold_signal - final_probs
    W2_grad = - np.outer(err_signal_output, hidden)
    err_signal_hidden = np.matmul(np.transpose(W2), err_signal_output)
    # err_signal_hidden_past_nonlin = err_signal_hidden * (1 - hidden ** 2)
    err_signal_hidden_past_nonlin = err_signal_hidden * ReLU_deriv(hidden)
    W1_grad = - np.outer(err_signal_hidden_past_nonlin, x)
    return W1_grad, W2_grad


def check_analytic_vs_empirical_gradient(x, y, W1, W2):
    """
    Lets you verify that your gradient computed analytically matches the "empirical gradient," the gradient obtained
    by changing a weight a small amount, computing the difference in likelihood, and dividing by the difference in
    the weight (i.e., a secant estimate of the loss change).
    :param x: input point
    :param y: label
    :param W1: weights
    :param W2: weights
    :return: None
    """
    delta = 0.0001
    hidden, final_probs = forward(x, W1, W2)
    loss_base = - np.log(final_probs[y])
    W1_grad, W2_grad = backward(x, y, hidden, final_probs, W2)
    for i in range(0, W1.shape[0]):
        for j in range(0, W1.shape[1]):
            delta_mat = np.zeros(W1.shape)
            delta_mat[i,j] += delta
            W1_new = W1 + delta_mat
            _, final_probs_perturbed = forward(x, W1_new, W2)
            emp_grad = (-np.log(final_probs_perturbed[y]) - loss_base)/delta
            print("W1 %i,%i: %f from empirical vs %f from analytic" % (i, j, emp_grad, W1_grad[i][j]))
    for i in range(0, W2.shape[0]):
        for j in range(0, W2.shape[1]):
            delta_mat = np.zeros(W2.shape)
            delta_mat[i,j] += delta
            W2_new = W2 + delta_mat
            _, final_probs_perturbed = forward(x, W1, W2_new)
            emp_grad = (-np.log(final_probs_perturbed[y]) - loss_base)/delta
            print("W2 %i,%i: %f from empirical vs %f from analytic" % (i, j, emp_grad, W2_grad[i][j]))


def sgd_test_nn(args):
    # Same data as in ffnn_example.py
    # Synthetic data for XOR: y = x0 XOR x1
    lr = args.lr
    epochs = args.epochs
    
    if args.func in alpha_map:
        lr, epochs = alpha_map[args.func]
    train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]], dtype=np.float32)
    train_ys = np.array([0, 1, 1, 1, 1, 0], dtype=np.int)
    # Inputs are of size 2
    input_vec_size = 2
    hidden_layer_size = 10
    num_output_classes = 2

    W1 = np.random.uniform(-0.1, 0.1, [hidden_layer_size, input_vec_size])
    W2 = np.random.uniform(-0.1, 0.1, [num_output_classes, hidden_layer_size])
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    for t in range(0, epochs):
        ll = 0.0
        correct = 0
        for idx in range(0, train_ys.shape[0]):
            [hidden, final_probs] = forward(train_xs[idx], W1, W2)
            if np.argmax(final_probs) == train_ys[idx]:
                correct += 1
            ll += -np.log(final_probs[train_ys[idx]])
            W1_grad, W2_grad = backward(train_xs[idx], train_ys[idx], hidden, final_probs, W2)
            W1 -= W1_grad * lr
            W2 -= W2_grad * lr
        print("Accuracy on epoch %i: %i/%i" % (t, correct, train_ys.shape[0]))
        print("Loss (negative log likelihood) on epoch %i: %f" % (t, ll))


if __name__ == '__main__':
    args = _parse_args()
    if args.func == "QUAD":
        sgd_test_quadratic(args)
    else:
        sgd_test_nn(args)
