from __future__ import print_function;

import numpy as np;
import scipy.io;

nPixels = 256;
nClasses = 10;

# nPixels = 4;
# nClasses = 2;

def test_gradient(model, data, wd_coefficient):
  base_theta = model_to_theta(model);
  h = 1e-2;
  correctness_threshold = 1e-5;
  analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient));
  # Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
  for i in range(100):
    test_index = ((i * 1299721) % base_theta.shape[0]) + 1; # 1299721 is prime and thus ensures a somewhat random-like selection of indices
    analytic_here = analytic_gradient[test_index];
    theta_step = base_theta * 0;
    theta_step[test_index] = h;
    contribution_distances = (-4, -3, -2, -1, 1, 2, 3, 4);
    contribution_weights = (1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280);
    temp = 0;
    for contribution_index in range(8):
      temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances[contribution_index]), data, wd_coefficient) * contribution_weights[contribution_index];
    fd_here = temp / h;
    diff = abs(analytic_here - fd_here);
    # fprintf('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
    if ( (diff >= correctness_threshold)
            | (diff / (abs(analytic_here) + abs(fd_here)) < correctness_threshold)
        ):
        raise Exception('Theta element #' + str(test_index)
                      + ', with value ' + str(base_theta[test_index])
                      + ', has finite difference gradient ' + str(fd_here)
                      + ' but analytic gradient ' + str(analytic_here)
                      + '. That looks like an error.');
  print('Gradient test passed. That means that the gradient that your code computed is within 0.001% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).');

def logistic(input):
    return( 1 / (1 + np.exp(-input)));

def log_sum_exp_over_rows(a):
    # # This computes log(sum(exp(a), 1)) in a numerically stable way
    maxs_small = np.amax(a, axis=0);
    maxs_small.shape = (1, maxs_small.shape[0]);
    maxs_big = np.tile(maxs_small, (a.shape[0], 1));
    return (np.log(np.sum(np.exp(a - maxs_big), 0)) + maxs_small);

def loss(model, data, wd_coefficient):
    # model['input_to_hid'] is a matrix of size <number of hidden units> by <number of inputs i.e. 256>. It contains the weights from the input units to the hidden units.
    # model['hid_to_class'] is a matrix of size <number of classes i.e. 10> by <number of hidden units>. It contains the weights from the hidden units to the softmax units.
    # data['inputs'][0][0] is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case.
    # data['targets'][0][0] is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.

    # Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units.
    hid_input = np.dot( model['input_to_hid'] , data['inputs'][0][0] ); # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input); # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    class_input = np.dot( model['hid_to_class'] , hid_output ); # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>

    # The following three lines of code implement the softmax.
    # However, it's written differently from what the lectures say.
    # In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
    # What we do here is exactly equivalent (you can check the math or just check it in practice), but this is more numerically stable.
    # "Numerically stable" means that this way, there will never be really big numbers involved.
    # The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
    # Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.

    class_normalizer = log_sum_exp_over_rows(class_input); # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1)); # log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
    class_prob = np.exp(log_class_prob); # probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>

    # select the right log class probability using that sum; then take the mean over all data cases.
    classification_loss = -np.mean(np.sum( np.multiply(log_class_prob , data['targets'][0][0]), 0));

    # weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    wd_loss = sum(np.power(model_to_theta(model),2)   )/2*wd_coefficient;

    return(classification_loss + wd_loss);

def d_loss_by_d_model(model, data, wd_coefficient):
    # model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
    # model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
    # data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case.
    # data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.

    # The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class. However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.

    # This is the only function that you're expected to change. Right now, it just returns a lot of zeros, which is obviously not the correct output. Your job is to replace that by a correct computation.
    input_to_hid = model['input_to_hid'] * 0;
    hid_to_class = model['hid_to_class'] * 0;

    return({'input_to_hid': input_to_hid, 'hid_to_class': hid_to_class});

def model_to_theta(model):
  # This function takes a model (or gradient in model form), and turns it into
  # one long vector. See also theta_to_model.
  input_to_hid_transpose = np.reshape(np.transpose(model['input_to_hid']), model['input_to_hid'].shape[0] * model['input_to_hid'].shape[1]);
  hid_to_class_transpose = np.reshape(np.transpose(model['hid_to_class']), model['hid_to_class'].shape[0] * model['hid_to_class'].shape[1]);
  return( np.concatenate((input_to_hid_transpose, hid_to_class_transpose)) );

def theta_to_model(theta):
  # This function takes a model (or gradient) in the form of one long vector
  # (maybe produced by model_to_theta), and restores it to the structure format,
  # i.e. with fields .input_to_hid and .hid_to_class, both matrices.
  n_hid = theta.shape[0] / (nPixels + nClasses);

  input_to_hid = np.transpose(np.reshape(   theta[range(nPixels * n_hid)]    ,    (nPixels, n_hid)    ));
  hid_to_class = np.reshape(theta[range(nPixels*n_hid, theta.shape[0])], (n_hid, nClasses)).T;

  return({'input_to_hid': input_to_hid, 'hid_to_class': hid_to_class});

def initial_model(n_hid):
    n_params = (nPixels + nClasses) * n_hid;
    as_row_vector = np.cos(np.array(range(n_params)));
    # We don't use random initialization, for this assignment.
    # This way, everybody will get the same results.
    return(theta_to_model(as_row_vector * 0.1));

def classification_performance(model, data):
    # This returns the fraction of data cases that is incorrectly classified by the model.

    # input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
    hid_input = np.dot(model['input_to_hid'] , data['inputs'][0][0]);
    # output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input);
    # input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
    class_input = np.dot( model['hid_to_class'] , hid_output);

    choices = np.argmax(class_input, 0); # choices is integer: the chosen class, plus 1.
    targets = np.argmax(data['targets'][0][0], 0); # targets is integer: the target class, plus 1.

    return(np.mean((choices != targets).astype(float)));

def a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size):
    model = initial_model(n_hid);
    datas = scipy.io.loadmat('data.mat')['data'];
    training = datas['training'][0][0];
    trainingInputs = training['inputs'][0][0];
    trainingTargets = training['targets'][0][0];
    n_training_cases = trainingInputs.shape[1];

    # print(model['input_to_hid'].shape);
    # print(model['hid_to_class'].shape);
    # print("############");
    # print("Number of training cases: " + str(n_training_cases));

    if n_iters != 0:
        test_gradient(model, training, wd_coefficient);

    # optimization
    theta = model_to_theta(model);
    momentum_speed = theta * 0;
    training_data_losses = [];
    validation_data_losses = [];
    if do_early_stopping:
        best_so_far = {   'theta': -1
                        , 'validation_loss': inf
                        , 'after_n_iters': -1
                      };
    for optimization_iteration_i in range(n_iters):
        model = theta_to_model(theta);

        training_batch_start = (((optimization_iteration_i-1) * mini_batch_size) % n_training_cases)+1;
        training_batch = {
                            'inputs':  trainingInputs[ :, training_batch_start : training_batch_start + mini_batch_size - 1]
                          , 'targets': trainingTargets[:, training_batch_start : training_batch_start + mini_batch_size - 1]
                         };
        gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient));
        momentum_speed = momentum_speed * momentum_multiplier - gradient;
        theta = theta + momentum_speed * learning_rate;

        model = theta_to_model(theta);
        training_data_losses = [training_data_losses, loss(model, datas.training, wd_coefficient)];
        validation_data_losses = [validation_data_losses, loss(model, datas.validation, wd_coefficient)];
        if do_early_stopping & (validation_data_losses[end] < best_so_far['validation_loss']):
          best_so_far = { 'theta':           theta # this will be overwritten soon
                        , 'validation_loss': validation_data_losses[end]
                        , 'after_n_iters':   optimization_iteration_i
                        };
        if optimization_iteration_i % round(n_iters/10) == 0:
          print(  'After ' + str(optimization_iteration_i)
                + ' optimization iterations, training data loss is ' + str(training_data_losses[end])
                + ', and validation data loss is ' + str(validation_data_losses[end]));
    if n_iters != 0:
        # Check again, this time with more typical parameters
        test_gradient(model, datas.training, wd_coefficient);
    if do_early_stopping:
        print('Early stopping: validation loss was lowest after ' + str(best_so_far['after_n_iters']) + ' iterations. We chose the model that we had then.');
        theta = best_so_far['theta'];
    # The optimization is finished. Now do some reporting.
    model = theta_to_model(theta);
    if n_iters != 0:
        # clf;
        # hold on;
        # plot(training_data_losses, 'b');
        # plot(validation_data_losses, 'r');
        # legend('training', 'validation');
        # ylabel('loss');
        # xlabel('iteration number');
        # hold off;
        pass;

    datas2 = (datas['training'][0][0], datas['validation'][0][0], datas['test'][0][0]);
    data_names = ('training', 'validation', 'test');
    for data_i in range(3):
        data = datas2[data_i];
        data_name = data_names[data_i];
        data_loss = loss(model, data, wd_coefficient);
        print('');
        print('The loss on the ' + data_name + ' data is ' + str(data_loss));
        if wd_coefficient!=0:
            data_loss_0 = loss(model, data, 0);
            print('The classification loss (i.e. without weight decay) on the ' + data_name + ' data is ' + str(data_loss_0));
        data_class_perf = classification_performance(model, data);
        print('The classification error rate on the ' + data_name + ' data is ' + str(data_class_perf));



if __name__ == '__main__':
    a3(0,0,0,0,0, False, 0);
