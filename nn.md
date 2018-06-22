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

# [Monotonic Networks](https://papers.nips.cc/paper/1358-monotonic-networks.pdf)
Authors propose new model that encodes the prior that learnt function has to be montone i.e. for increasing input the output 
cannot be dicreasing. The prior is necessary to make learning efficient. Authors test their model on the prediction of bond rating and achieve 
better results that previous methods.
The inputs are first transformed linearly to multiple grops by using constraint weights which are positive for increasing
monotonicity and negative for decreasing monotonicity (weight constraints are enforced by using exp of free parameter).
The gropus are processed by max operator and then by min operator.
The whole model can be learned by the gradient descent.