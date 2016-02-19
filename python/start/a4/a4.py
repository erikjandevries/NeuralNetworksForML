from __future__ import print_function;

import numpy as np;
import scipy.io;

def describe_matrix(matrix):
    print('Describing a matrix of size ' + str(matrix.shape[0]) + ' by ' + str(matrix.shape[1]) + '. '
        + 'The mean of the elements is ' + str(np.mean(matrix)) + '. '
        + 'The sum of the elements is ' + str(np.sum(matrix)) + '.');

def classification_phi_gradient(input_to_class, data):
    # This is about a very simple model: there's an input layer, and a softmax output layer. There are no hidden layers, and no biases.
    # This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
    # <input_to_class> is a matrix of size <number of classes> by <number of input units>.
    # <data> has fields .inputs (matrix of size <number of input units> by <number of data cases>) and .targets (matrix of size <number of classes> by <number of data cases>).
    # first: forward pass
    class_input = input_to_class * data.inputs; # input to the components of the softmax. size: <number of classes> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input); # log(sum(exp)) is what we subtract to get normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1)); # log of probability of each class. size: <number of classes> by <number of data cases>
    class_prob = np.exp(log_class_prob); # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes> by <number of data cases>
    # now: gradient computation
    d_loss_by_d_class_input = -(data.targets - class_prob) / data.inputs.shape[1]; # size: <number of classes> by <number of data cases>
    d_loss_by_d_input_to_class = d_loss_by_d_class_input * np.transpose(data.inputs); # size: <number of classes> by <number of input units>
    d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class;
    return(d_phi_by_d_input_to_class);

def argmax_over_rows(matrix):
    return(np.argmax(matrix, axis=0));

def log_sum_exp_over_rows(a):
    # # This computes log(sum(exp(a), 1)) in a numerically stable way
    maxs_small = np.amax(a, axis=0);
    maxs_small.shape = (1, maxs_small.shape[0]);
    maxs_big = np.tile(maxs_small, (a.shape[0], 1));
    return (np.log(np.sum(np.exp(a - maxs_big), 0)) + maxs_small);

def logistic(input):
    return(1 / (1 + np.exp(-input)));

class dataset(object):
    def __init__(self, inputs, targets):
        self.inputs = inputs;
        self.targets = targets;

class a4(object):
    def __init__(self):
        self.randomness_source = scipy.io.loadmat('a4_randomness_source.mat')['randomness_source'][0];
        datas = scipy.io.loadmat('data_set.mat')['data'];
        training = datas['training'][0][0];
        validation = datas['validation'][0][0];
        test = datas['test'][0][0];
        self.data_sets = {  'training':   dataset(  training['inputs'][0][0]
                                                  , training['targets'][0][0] )
                          , 'validation': dataset(  validation['inputs'][0][0]
                                                  , validation['targets'][0][0] )
                          , 'test':       dataset(  test['inputs'][0][0]
                                                  , test['targets'][0][0] )
                         };

        self.report_calls_to_sample_bernoulli = False;
        # self.report_calls_to_sample_bernoulli = True;

        self.test_rbm_w       = self.rand((100, 256), 0) * 2 - 1;
        self.small_test_rbm_w = self.rand(( 10, 256), 0) * 2 - 1;

        temp = self.extract_mini_batch(self.data_sets['training'], 1, 1);
        self.data_1_case = self.sample_bernoulli(temp.inputs);
        temp = self.extract_mini_batch(self.data_sets['training'], 100, 10);
        self.data_10_cases = self.sample_bernoulli(temp.inputs);
        temp = self.extract_mini_batch(self.data_sets['training'], 200, 37);
        self.data_37_cases = self.sample_bernoulli(temp.inputs);

        self.test_hidden_state_1_case = self.sample_bernoulli(self.rand((100,1), 0));
        self.test_hidden_state_10_cases = self.sample_bernoulli(self.rand((100,10), 1));
        self.test_hidden_state_37_cases = self.sample_bernoulli(self.rand((100,37), 2));

        self.report_calls_to_sample_bernoulli = True;

    def rand(self, requested_size, seed):
        start_i = (round(seed) % round(self.randomness_source.shape[0] / 10) );
        # print(start_i + 1);
        if start_i + np.prod(requested_size) >= self.randomness_source.shape[0]:
            throw("a4.rand failed to generate an array of that size (too big)");
        end_i = start_i + np.prod(requested_size);
        return(np.reshape(self.randomness_source[start_i:end_i], requested_size, order='F'));

    def extract_mini_batch(self, data_set, start_i, n_cases):
        inputs  = data_set.inputs[:,  (start_i-1):(start_i + n_cases -1)];
        targets = data_set.targets[:, (start_i-1):(start_i + n_cases -1)];
        return(dataset(inputs, targets));

    def sample_bernoulli(self, probabilities):
        if self.report_calls_to_sample_bernoulli:
            print('sample_bernoulli() was called with a matrix of size ' + str(probabilities.shape[0])
                + ' by ' + str(probabilities.shape[1]) + '.');
        seed = np.sum(probabilities);
        return((probabilities > self.rand(probabilities.shape, seed)).astype(int));

    def main(self, n_hid, lr_rbm, lr_classification, n_iterations):
        self.report_calls_to_sample_bernoulli = False;

        # first, train the rbm
        rbm_w = self.optimize(  [n_hid, 256]
                              , lambda (rbm_w, data): self.cd1(rbm_w, data.inputs) # discard labels
                              , self.data_sets['training']
                              , lr_rbm
                              , n_iterations);
        # rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>

        self.show_rbm(rbm_w);

        input_to_hid = rbm_w;

        # calculate the hidden layer representation of the labeled data
        hidden_representation = logistic(np.dot(input_to_hid , self.data_sets['training'].inputs));

        # train hid_to_class
        data_2 = dataset(hidden_representation, self.data_sets['training'].targets);
        hid_to_class = self.optimize(  [10, n_hid]
                                     , lambda (model, data): classification_phi_gradient(model, data)
                                     , data_2
                                     , lr_classification
                                     , n_iterations);

        # report results
        data_names = ('training', 'validation', 'test');
        datas = (self.data_sets['training'], self.data_sets['validation'], self.data_sets['test']);

        # for data_name, data in self.data_sets.iteritems():
        for data_name in ('training', 'validation', 'test'):
            data = self.data_sets[data_name];
            hid_input = np.dot(input_to_hid , data.inputs); # size: <number of hidden units> by <number of data cases>
            hid_output = logistic(hid_input); # size: <number of hidden units> by <number of data cases>
            class_input = np.dot(hid_to_class , hid_output); # size: <number of classes> by <number of data cases>
            class_normalizer = log_sum_exp_over_rows(class_input); # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
            log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1)); # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
            error_rate = np.mean((argmax_over_rows(class_input) != argmax_over_rows(data.targets)).astype(float)); # scalar
            loss = -np.mean(sum(log_class_prob * data.targets, 0)); # scalar. select the right log class probability using that sum; then take the mean over all data cases.
            print('For the ' + str(data_name) + ' data, the classification cross-entropy loss is ' + str(loss) +
                  ', and the classification error rate (i.e. the misclassification rate) is ' + str(error_rate));

        self.report_calls_to_sample_bernoulli = True;

    def optimize(self, model_shape, gradient_function, training_data, learning_rate, n_iterations):
        # This trains a model that's defined by a single matrix of weights.
        # <model_shape> is the shape of the array of weights.
        # <gradient_function> is a function that takes parameters <model> and <data> and returns the gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing. Note the contrast with the loss function that we saw in PA3, which we were minimizing. The returned gradient is an array of the same shape as the provided <model> parameter.
        # This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
        # This returns the matrix of weights of the trained model.
        model = (self.rand(model_shape, np.prod(model_shape)) * 2 - 1) * 0.1;
        momentum_speed = np.zeros(model_shape);
        mini_batch_size = 100;
        start_of_next_mini_batch = 1;
        for iteration_number in range(n_iterations):
            mini_batch = self.extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);
            start_of_next_mini_batch = (start_of_next_mini_batch + mini_batch_size) % training_data.inputs.shape[1];
            gradient = gradient_function(model, mini_batch);
            momentum_speed = 0.9 * momentum_speed + gradient;
            model = model + momentum_speed * learning_rate;
        return(model);

    def show_rbm(self, rbm_w):
        n_hid = rbm_w.shape[0];
        n_rows = ceil(sqrt(n_hid));
        blank_lines = 4;
        distance = 16 + blank_lines;
        to_show = np.zeros((n_rows * distance + blank_lines, n_rows * distance + blank_lines));
        for i in range(n_hid):
            row_i = floor(i / n_rows);
            col_i = i % n_rows;
            # pixels = np.transpose(np.reshape(rbm_w[i, :], (16, 16)));
            pixels = np.reshape(rbm_w[i, :], (16, 16));
            row_base = row_i*distance + blank_lines;
            col_base = col_i*distance + blank_lines;
            to_show[row_base:row_base+16, col_base:col_base+16] = pixels;
        # extreme = np.amax(np.absolute(to_show));
        plt.imshow(to_show);
        plt.title('hidden units of the RBM');
        plt.axis('off')
        plt.show();

    def visible_state_to_hidden_probabilities(self, rbm_w, visible_state):
        # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        # <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
        # The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
        # This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
        raise Exception("not yet implemented");

    def hidden_state_to_visible_probabilities(self, rbm_w, hidden_state):
        # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        # <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
        # The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
        # This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.
        raise Exception("not yet implemented");

    def configuration_goodness(self, rbm_w, visible_state, hidden_state):
        # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        # <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
        # <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
        # This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
        raise Exception("not yet implemented");

    def configuration_goodness_gradient(self, visible_state, hidden_state):
        # <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
        # <hidden_state> is a (possibly but not necessarily binary) matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
        # You don't need the model parameters for this computation.
        # This returns the gradient of the mean configuration goodness (negative energy, as computed by function <configuration_goodness>) with respect to the model parameters. Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to this function. Notice that we're talking about the mean over data cases (as opposed to the sum over data cases).
        raise Exception("not yet implemented");

    def cd1(self, rbm_w, visible_data):
        # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
        # <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
        # The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
        raise Exception("not yet implemented");


if __name__ == '__main__':
    myA4 = a4();
    myA4.main(300, 0, 0, 0);
