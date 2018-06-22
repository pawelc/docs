# [Efficient Gradient-Based Inference through Transformations between Bayes Nets and Neural Nets](https://arxiv.org/abs/1402.0480)
Authors show new way of inference in probabilistic models.
When hidden variables are strongly dependant on each other than the posterior is ill-conditioned. This makes inference less
efficient because of smaller optima step size in gradient based optimization or sampling methods.
Authors show how to convert model with probabilistic continuous hidden variables (Centered Parametrization) to the model
with deterministic variables (Differentiable not Centered Parametrization) by adding auxiliary variables and using Jackobian
of the function in DNCP.
There are different ways of converting CP to DNCP like: tractable and differentiable inverse of CDF, location-scale parametric family 
with auxiliary variable, composition.
Using this approach the dropout network can be seen as bayesian network.
Using this parametrization we obtain differentiable MC likelihood estimator that can be used to learn parameters. 
TODO: Read it later because too time consuming for now.