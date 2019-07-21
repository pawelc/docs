# A statistical view of deep learning 
DNN = Recursive Generalized Linear Models
Autoencoders as Variational approach to learning generative models.
RNN as state space models with deterministic dynamics.
Maximum likelihood, MAP and overfitting. But they lack transformation invariance.
Thoughts when model is deep.

# A neural network is a monference, not a model
NN as model with inference. Shows that RNN directly recovere inference of HMM (filtering) with the same precision. The same as in 
"Probabilistic Interpretations of Recurrent Neural Networks"

# Uncertainty in Deep Learning
Mapping VI with Stochastic Regularization Techniques like dropout. Should  be implementable for DNC.
There are a lot of references in this work worth reading. It would be nice to read later to the end especially about usage 

# [Multilayer Feedforward Networks are Universal Approximators]
The rigorous proof using advanced math of the NN being universal approximators on Borel measurable functions.

#[Training a 3-node neural network is NP-complete]()
Authors prove that training network with 2 hidden neurons and 1 output neuron is NP-Complete by showing that it is equivalent 
to the set spitting problem which is NP-Complete. They also show that using more complex structure problem can become of class P.
The problem here is classification of binary input and the question is only if the set of weight exists to correctly classify
training data. Finding the parameters is at least as difficult problem. Not read the full proof.

# [Monotonicity Hints](https://papers.nips.cc/paper/1270-monotonicity-hints.pdf)
1997
Authors derive cost function so the NN can meet monotonicity constraint in a soft way by penalizing cost function if not met.
So this is different that forcing monotonicity by NN design. They use Bayesian formula to introduce additional term. They fit
average of the data so it won't work with multimodal data. 
First stage the NN is trained using only maximum likelihood criterion (minimize MSE) then additionally they add approximate for the monotonicity
constraint (monotonicity error penalty for the rest of the training). From the experiments it looks that this monotonicity
criteria is needed to obtain good performance. The NN model without monotonicity hint could be even weaker than linear model when
training goes into overfitting. Therefore we also should use validation set for early stopping.

# [Efficient backprop]()
1998
A lot of ideas how to improve training using gradient optimization. Like stochastic training, presenting examples
with the most information content (obtaining the biggest errors for example), increasing batch size when training
progresses, normalizing the inputs at each layer, keeping scale of less important inputs smaller, using tanh instead 
of vanilla sigmoid (outputs will be centered around 0), adding linear term to tanx to avoid flat spots, extending the 
range of the tanh so the labels do not fall into its saturation points, initialize weight in a way the non-linearities
operate in their linear range, if weights are too small then gradients will be also small, authors give recipe for the
weight initialization so the following layers are normalized in the beginning of training, the 2nd derivative of the loss
with respect to the weights of the lower levels is usually smaller than upper levels therefore learning rates of weights
at lower levels should be larger,  adaptive learning rate, using RBF units, we can diagonalize hessian (transforming) the 
parameters so we can use separate update rates for each parameter, hessian matrix is not positive definite everywhere
in NN so the standard Newton algorithm is not appropriate here, conjugate gradients (linear in number of weights, does
not use Hessian explicitly, works only for batch learning), quasi newton (BFGS: iteratively computes H^-1, O(N), uses 
line search, only for batch learning), Gauss Newton and Levenberg Marquardt (use the square Jacobi approximation, mainly
for batch learning, O(N^3), work only MSE loss functions), Tricks to compute the hessian matrix efficiently in NN (finite 
difference method) 
We see that many ideas were recently reinvented. 

# [Monotonic Networks](https://papers.nips.cc/paper/1358-monotonic-networks.pdf)
1998

**Min/Max networks**
Authors propose new model that encodes the prior that learnt function has to be monotone i.e. for increasing input the output 
cannot be decreasing. The prior is necessary to make learning efficient. Authors test their model on the prediction of bond rating and achieve 
better results that other models. They compare it to linear model and NNs with different sizes. They find that NN trained to achieve
best result on the training data achieves worse result on test data than linear model. Using validation data and early stopping makes NN better
than linear model but still worse than monotone.
The inputs are first transformed linearly to multiple groups by using constraint weights which are positive for increasing
monotonicity and negative for decreasing monotonicity (weight constraints are enforced by using exp of free parameter).
The groups are processed by max operator and then by min operator.
The whole model can be learned by the gradient descent.
The max operator allows to model convex part of monotone function and min allows to model concave part. Authors prove that their model
can approximate any continuous and with finite first partial derivative functions to a desired accuracy.
Authors notice that swapping min and max operator would also work. And leave it for future research to find out why/when to use which.
Also given model has few hyper-parameters that has to be decided like: number of groups or number of hyper-planes within the group.
  
# [Feedforward networks with monotone constraints](https://ieeexplore.ieee.org/document/832655/)
1999.07.10

Authors use **NN with all positive weights** by applying transformation to the weight without constraint exp(w) so they 
can approximate **all monotonic functions**. They should that this constraint networks even without using early stopping 
can achieve comparable generalization to the generic NN with early stopping. They notice that the networks they checked can have
limitations because activations used have strictly positive derivatives which can make it impossible to model functions like x^3.

# [Monotonic multi-layer perceptron networks as universal approximators](https://link.springer.com/chapter/10.1007/11550907_6)
2005

Again authors use NN with 2 hidden layers. They prove that if the output and hidden to hidden weights are positive we can set
constrain input to hidden layers weights to be positive/negative depending on monotonicity constraints. If we don't want to 
enforce given input to be monotone with output we dont constrain its input to hidden weight. They also notice that min/max
networks tend to be more expensive to train then this architecture. They empirically show that monotonic NN with 2 hidden 
layers achieves comparable error on the training data and much better on the test data when there is requirement for the monotonicity
compared to regular NN with 1 hidden layer.

# [Comparison of universal approximators incorporating partial monotonicity by structure](https://www.sciencedirect.com/science/article/pii/S0893608009002330)
2009.02.20

Article reviews "Monotonic Networks" and "Feedforward networks with monotone constraints". They perform test on multiple data sets
which have exhibit monotonicity on some of the inputs. They use MSE and R-squared to compare the models.
Authors conclude based on tests that both network excels in different areas and their application depends on the problem.

# [Monotone and Partially Monotone Neural Networks](https://ieeexplore.ieee.org/document/5443743/)
2010.06.06

Authors prove monotone behaviour of 2 hidden layer min-max networks. They prove that any multivariate function with k 
variables can be approximated with NN with at most k hidden layers and positive weights. They propose heuristic method 
to determine monotonicity of individual independent variables. They demonstrate that partially monotone NN achieve smaller
errors and variance than unconstrained NNs. I haven't read the paper 100% because a lot of proofs which I don't need at the momemnt.
Although there are some interesting results about partial monotonicity which could be beneficial for conditional modeling.

# [Understanding the difficulty of training deep feedforward neural networks]()
2010



# [Bounded activation functions for enhanced training stability of deep neural networks on visual pattern recognition problems]()
2016

Mention that tanh is better than logistic but still has large saturation regions. Authors provide conditions for
non linearity so it can be used in universal approximator. Authors present bounded versions of most unbounded 
nonlinearities like ReLU, bifiring neuron, leaky ReLU
