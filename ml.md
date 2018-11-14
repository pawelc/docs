# [Deep Learning: Autodiff, Parameter Tying and Backprop Through Time](http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/ParameterTying.pdf)
2015.02.09

Overview of different ways of differentiation: numerical, symbolic and autodiff. Report present how to efficiently compute
gradient with respect to tied parameters like in RNN. Examples showing autodiff and parameter tying.

# [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
2018.02.05

Good survey of AD methods. How they refer to other methods like backpropagation. History. Application in different areas
of machine learning. Summarize in my thesis.

# Deep Exponential Families
Using different distributions from exponential families  for latent variables we can recover different models.
 Bernoulli latent variables recover the classical sigmoid belief network
Gamma  latent variables give something akin to deep version of nonnegative matrix factorization
Gaussian latent variables lead to the types of models that have recently been explored in the context Deep Exponential Families of computer vision
Using sparse gamma (shape less than 1) to model variables and weights
The explaining away makes inference harder in DEFs then in RBMs but forces a more parsimonious representation where similar features compete to explain the data rather than work in tandem (possibly read  [12, 1] references)
Somewhat related, we find sigmoid DEFs (with normal weights) to be more difficult to train and deeper version perform poorly (worth to check why)