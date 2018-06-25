# [Monotonic Networks](https://papers.nips.cc/paper/1358-monotonic-networks.pdf)
1998

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

# [Density estimation and random variate generation using multilayer networks](https://ieeexplore.ieee.org/abstract/document/1000120/)
2002.08.07

Authors describe 2 new methods for pdf estimation using NN. Paper contains nice review of literature of density estimation using
standard methods and NNs. Authors finds bound on estimation error and prove it and compare to other methods.
The first method uses NN do model distribution function of the data. The input to the NN is data. We know that for NN to be CDF
it has to generate uniform data. So for input data we generate uniform data as target. Training NN using supervised learning for input data and 
generated uniform data and constraining output to be monotone we get CDF. Differentiating this NN wrt input gives us density function.
The second method creates training data by taking X to be data to be modelled and Y is the CDF estimated by the MC computation of the integral for CDF using data.
Then pdf can be computed as before i.e. taking derivative of CDF network wrt Xes.
This methods compute distribution function for data and not for conditional/predictive distribution.
Random data generation:
They also propose few methods to to generate random data from learned distribution function by learning the 2nd NN as the inverse of CDF.
(It is kind of auto-encoder where middle layer learn to produce the uniform distribution)
Authors also notice that to train network generating the data we don't have to train first distribution function but we can learn straight away
inverse CDF and then we can sample new data. This is done by 2 analogous methods as described before. For example by feeding uniform data to network
and expecting X as output (of course before we have to sort Xes and uniform input).
Also we can generate random data given pdf by generating training data for NN by doing numerical integration of pdf. Then again by feeding
uniform data to this network we can generate random data from this network.
One more method to generate random data is named as control method and it consist of 2 NNs, first input some standard distribution, 1st NN
outputs target data and 2nd NN produces distribution function taking input from the 1st NN output. 2nd NN is learned using one of the previous methods
to learn CDF. This CDF is compared to the true CDF and the error signal is passed to the 1st NN. This is repeated over many iterations.
This way we can learn to generate multivariate RV. 
Authors test their algorithms on true financial data and on generated data. They compare their methods to kernel density estimation.
Very mathy and it would take long to read everything. But nice examples so possibly nice to reread if doing similar experiments.

# [Consistent Density Function Estimation with Multilayer Perceptrons](http://ieeexplore.ieee.org.libproxy.ucl.ac.uk/document/1716228/)
2006.07.16

Authors show nice theoretical grounds how to train NN to model distribution function. They show different situations and maths for it.
First when we know target pdf we can use simply relative entropy i.e. KL-divergence. They show how KL-divergence split into
entropy of the function and performance index which is cross entropy between true and model distributions. Secondly when distribution is not known
but we have only data we use MC approximation which gives the same as ML. Thirdly they show density estimation from the perspective of
maximum entropy. Here the output of the trnasformation has to be constrained by (0,1) cube. 
First and second methods (practical one) match i.e. empirical performance index accurately represents the performance index then it
is consistent. Authors use iterative procedure to check consistency by adding new sampples to the training test and checking the 
change in performance index. If it doesn't change for some defined number of iterations we can assume that we arrived at
consistent estimate. We can compare the value of the empirical performance index to differential entropy of the true distribution to 
check the training performance quantitatively.
Authors notice that pdf spikes in the area of high probability areas comes from the log in the performance index because these 
areas are less penalized than tail regions. Authors improve results by employing simpler NNs and more data but notice that it was done
because knowledge of the modelled density.
When they apply this method directly to the estimation of normalized Gaussian the outcome is very wiggly pdf function. This happens
because of the poor local minima and flexibility of NN. Authors try to use more data but it also doesn't help. One problem
is that training process doesn't focus on monotonically increasing functions.
To get monotonically increasing functions authors propose constraining the weights to be positive and output being sigmoid without
a bias. This looks again like fail but comparing it to data histogram it looks it fits quite well.
By using again more data we recover proper pdf.
They prove the determinant of the Jackobian of the transformation is equal to the sought input density.
They equal it to the previous performance index.
Using NN with hidden layer of size at least as the input/output to increase probability that learnt function is invertible.

# [Efficient Gradient-Based Inference through Transformations between Bayes Nets and Neural Nets](https://arxiv.org/abs/1402.0480)
2014.02.03
 
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

[Soft-Constrained Nonparametric Density Estimation with Artificial Neural Networks](https://link.springer.com/chapter/10.1007/978-3-319-46182-3_6)
2016.09.09

NNs are promising to learn pdf because its universal approximator property. One of the problem is that we have to constrain the pdf
to integrate to 1. Authors mention it is harder to learn pdf directly than it is to train NN to recover CDF and then differentiate
to recover pdf. **Authors notice that good approximation of CDF doesn't have translate to good approximation of pdf**. For example small
error between true and learnt CDF doesn't mean for example large fluctuations of learnt CDF which can cause rapidly changing derivative.
The other problem is that negative derivatives can occur and integration to 1 is also not accounted for. Authors mention
their previous paper using Perzen window but this algo doesn't emply any window. THis model uses empirical estimation of modeled
pdf. Authors use MCMC method with NN and optimization to enforce integration to one of the pdf.
Authors use logistic sigmoid with addaptive amplitude in output so output is (0,+inf) i.e. \lambda/(1+exp(a_i)).
They train NN that minimizes specially designed cost function composed of 2 parts: matching empirical probability 
and integral over whole pdf is 1. The second term is multiplies by penalty term which has to be chosen.
To compute the target for pdf they use 2 facts: how to compute probability in the ball geometrically and using frequentist stats, from there they have:
p(x) = (k_n/n)/V(B(x,T)). 
To compute the gradient of the cost function algorithm has to compute also 2 integrals over whole input space (which is compact by
assumption). Authors use MC approximation using importance sampling using mixture of uniform and pdf given by learnt model. Sampling
from the current model (given by the NN) is done using Metropolis-Hastings algorithm with proposal given by multivariate logistic pdf.
Therefor the method seems quite resource consuming.
As the test set they use mixture of 3 Fisher-Tippet distributions (Generalized extreme value distribution distributions). This distributions
show asymmetries like positive skewness so can be difficult to model by estimators with symmetric kernels.
Proposed model is compared with k-NN (which outputs not normalized pdf and is unbounded) and Parzen Window (which does not model skewness 
aspect). The proposed model approximate distribution was much better: it captured skewness, numerically checked integral of pdf
was very close to 1 and was very smooth.
Authors want to present in the future work this method extended to multidimensional data. Actually this is done in:
"Soft-Constrained Neural Networks for Nonparametric Density Estimation".

# [Density transformation and parameter estimation from back propagation algorithm](https://ieeexplore.ieee.org/document/7727184/)
2016.10

Method for training NN to **transform one pdf into another**. It also gives estimate on the **number of hidden neurons** that depend
on the statistics of the input data. Authors use 2 NNs, one to transform one distribution to the uniform and 2nd NN to transform
uniform to target density. They deduce this approach from the well known fact in statistics that when data of given distribution is 
passed through CDF it gives uniform and when uniform is passed through inverse of CDF the samples are from the CDF.
They revisit the work in "Density estimation and random variate generation using multilayer networks".
The strongest part of this work is mathematical derivation of the bound of number of hidden neurons needed for given 
approximation but haven't checked the derivation.

# [From CDF to PDF --- A Density Estimation Method for High Dimensional Data](https://arxiv.org/abs/1804.05316)
2018.04.15

Authors improve on method described in "Density estimation and random variate generation using multilayer networks". 
They improve upon this method by **making the output monotone** (so no hyperparameter tuning is necessary) increasing 
(by constraining the weight to be positive) and devise algorithm to compute high order derivative to compute pdf from the CDF (by using
someone else algorithm  to determine coefficients of the polynomial, we need it to compute n-th derivative of tgh which is needed
to compute **higher order derivative**). Authors also notice that using sigmoid functions can be inefficient for approximating
strong non-smoothness like uniform distribution. They use combination of tanh and -1,x,1 function. They made also combination
parametrized to it can be fine tuned by running optimization.