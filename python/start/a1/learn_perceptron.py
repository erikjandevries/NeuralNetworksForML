import numpy as np
import scipy.io

from plot_perceptron import *

dataset = scipy.io.loadmat('../data/dataset1.mat')
#dataset = scipy.io.loadmat('../data/dataset2.mat')
#dataset = scipy.io.loadmat('../data/dataset3.mat')
#dataset = scipy.io.loadmat('../data/dataset4.mat')

w_gen_feas          = dataset['w_gen_feas']
w_init              = dataset['w_init']
pos_examples_nobias = dataset['pos_examples_nobias']
neg_examples_nobias = dataset['neg_examples_nobias']


def add_col_ones(data):
    return np.hstack((data, np.ones((data.shape[0],1))))

def error_report(i, num_errs, w, w_gen_feas): 
    errstr = 'Number of errors in iteration %d:\t%d\n' % (i, num_errs)
    weightstr = '\tweights=%s\n' % str(w)
    if(len(w_gen_feas) == 0):
        feasiblestr = ''
    else:
        dist = np.linalg.norm(w - w_gen_feas)
        feasiblestr = '\tdistance to feasible=%.3f' % dist
    return errstr + weightstr + feasiblestr
    
def learn_perceptron(neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas):
    """Learns the weights of a perceptron for a 2-dimensional dataset and plots
    the perceptron at each iteration where an iteration is defined as one
    full pass through the data. If a generously feasible weight vector
    is provided then the visualization will also show the distance
    of the learned weight vectors to the generously feasible weight vector.
    Required Inputs:
      neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0.
          num_neg_examples is the number of examples for the negative class.
      pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1.
          num_pos_examples is the number of examples for the positive class.
      w_init - A 3-dimensional initial weight vector. The last element is the bias.
      w_gen_feas - A generously feasible weight vector.
    Returns:
      w - The learned weight vector.
    """

    # Bookkeeping
    num_neg_examples = neg_examples_nobias.shape[0]
    num_pos_examples = pos_examples_nobias.shape[0]
    num_err_history = []
    w_dist_history = []

    # Here we add a column of ones to the examples in order to allow us to learn
    # bias parameters.
    neg_examples = add_col_ones(neg_examples_nobias)
    pos_examples = add_col_ones(pos_examples_nobias) 

    # If weight vectors have not been provided, initialize them appropriately.
    if (w_init is None):
        w = NP.random.randn(neg_examples.shape[1],1)
    else:
        w = w_init

    if (w_gen_feas is None):
        w_gen_feas = []
    
    # Find the data points that the perceptron has incorrectly classified
    # and record the number of errors it makes.
    iteration = 0;
    (mistakes0, mistakes1) = eval_perceptron(neg_examples, pos_examples, w)
    num_errs = len(mistakes0) + len(mistakes1)
    num_err_history.append(num_errs)
    print error_report(iteration, num_errs, w, w_gen_feas)

    plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)
    key = raw_input('<Press enter to continue, q to quit.>')
    if key == 'q':
        return w

    # If a generously feasible weight vector exists, record the distance
    # to it from the initial weight vector.
    if len(w_gen_feas) != 0:
        w_dist_history.append(np.linalg.norm(w - w_gen_feas))
    
    # Iterate until the perceptron has correctly classified all points.
    while (num_errs > 0 and iteration < 1000):
        iteration += 1;
    
        # Update the weights of the perceptron.
        w = update_weights(neg_examples, pos_examples, w);
    
        # If a generously feasible weight vector exists, record the distance
        # to it from the current weight vector.
        if len(w_gen_feas) != 0:
            w_dist_history.append(np.linalg.norm(w - w_gen_feas))
    
        # Find the data points that the perceptron has incorrectly classified.
        # and record the number of errors it makes.
        (mistakes0, mistakes1) = eval_perceptron(neg_examples, pos_examples, w)
        num_errs = len(mistakes0) + len(mistakes1)
        num_err_history.append(num_errs)
    
        print error_report(iteration, num_errs, w, w_gen_feas)

        plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history);
        key = raw_input('<Press enter to continue, q to quit.>')
        if (key == 'q'):
            break;

    return w

def update_weights(neg_examples, pos_examples, w_current):
    """Updates the weights of the perceptron for incorrectly classified points
    using the perceptron update algorithm. This function makes one sweep
    over the dataset.
    Inputs:
      neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
          num_neg_examples is the number of examples for the negative class.
      pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
          num_pos_examples is the number of examples for the positive class.
      w_current - A 3-dimensional weight vector, the last element is the bias.
    Returns:
      w - The weight vector after one pass through the dataset using the perceptron
          learning rule.
    """

    w = w_current.copy()

    for (i, neg_example) in enumerate(neg_examples):
        activation = np.dot(neg_example, w)
        if(activation >= 0):
            # YOUR CODE HERE
            pass
            
    for (i, pos_example) in enumerate(pos_examples):
        activation = np.dot(pos_example, w)
        if(activation < 0):
            # YOUR CODE HERE
            pass
            
    return w

def eval_perceptron(neg_examples, pos_examples, w):
    """Evaluates the perceptron using a given weight vector. Here, evaluation
    refers to finding the data points that the perceptron incorrectly classifies.
    Inputs:
      neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
          num_neg_examples is the number of examples for the negative class.
      pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
          num_pos_examples is the number of examples for the positive class.
      w - A 3-dimensional weight vector, the last element is the bias.
    Returns:
      mistakes0 - A vector containing the indices of the negative examples that have been
          incorrectly classified as positive.
      mistakes0 - A vector containing the indices of the positive examples that have been
          incorrectly classified as negative.
    """
    
    mistakes0 = [];
    mistakes1 = [];

    for (i, neg_example) in enumerate(neg_examples):
        activation = np.dot(neg_example, w)
        if (activation >= 0):
            mistakes0.append(i)

    for (i, pos_example) in enumerate(pos_examples):
        activation = np.dot(pos_example, w)
        if (activation < 0):
            mistakes1.append(i)

    return (mistakes0, mistakes1)





w = learn_perceptron(neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas)

