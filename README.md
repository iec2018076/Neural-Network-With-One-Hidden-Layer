# Neural-Network-With-One-Hidden-Layer
In this Project a 2-class classification neural network with a single hidden layer is implemented. Using units with a non-linear activation function, such as tanh(hyper-tan). Then computing the cross entropy loss followed by the forward and backward propagation used to update the parameter using the Gradient Descent Algorithm .

- a numpy-array (matrix) X that contains your features (x1, x2)
- a numpy-array (vector) Y that contains your labels (red:0, blue:1).

## shape
The shape of X is: (2, 200)
The shape of Y is: (1, 200)
I have m = 200 training examples!

First we use the Logistic Regression Model from sklearn library to test the accuaracy

## Accuracy of Logistic Reggression 
Accuracy of logistic regression: 50 % (percentage of correctly labelled datapoints)
Interpretation: The dataset is not linearly separable, so logistic regression doesn't perform well. Hopefully a neural network will do better.

# Some Important Function Description :

## forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
## compute_cost(A2, Y):
    """    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation
    """
 
## backward_propagation(parameters, cache, X, Y):
    """  
    Arguments:
    parameters -- python dictionary containing parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    
## update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent 
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    
## Hyper-Parameter Tuning 
We can take tune hidden_layer_sizes  by taking a range of values and check the accuracy for it and choose the best fit 
eg: for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size)*100)
    
Similarly We can tune learning_rate (alpha) and 
### choice of the activation function 
1) sigmoid      a= 1/(1+pow(e,-z))
2) hyper-tan    a= (pow(e,z)-pow(e,-z))/(pow(e,z)+pow(e,-z))
3) ReLu         a= max(0,z)
4) leaky ReLu   a= max(0.01z,z)

