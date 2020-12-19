# [Probabilistic Neural Networks]()
1989

Authors use NN that computes something along the Parzen density estimators. Therefore it outputs the density function
directly. The hidden nodes (here called the pattern units) uses exp function (instead of sigmoid) so they represent 
components of the mixture (here we have one pattern unit for each training pattern/example). The following layer sums 
the pattern units (there is one unit is the summation layer per category to classify). 
The output unit performs Bayesian decision (taking into consideration prior probabilities of class
and loss of classifying incorrectly). The model is parametric because is stores all training examples as pattern nodes.
The sigma parameter controls the smoothness of the estimator, sigma -> 0 then classifier is nearest neighbour and when
sigma -> inf the hyperplane.

# [Probabilistic neural networks and the polynomial adaline as complementarp techniques for classification]()
1990

Here authors review Probabilistic Neural Networks (requires to store all training points) and polynomial methods 
(do not) which are better than backprop NN (according to authors) because they require one pass through the data to train. 
The PNN is for smaller data sets and polynomial can be applied to bigger data sets. The decision surface of the 
PNN approaches Bayes optimal one with the increase of the training examples.  Authors provide equations for the polynomial
with the same decision boundary as PNN which coefficients are given by analytical formulas.

# [Integrating Probabilistic Rules into Neural Networks: A Stochastic EM Learning Algorithm]()
1991

Authors show how to integrate probabilistic inference network (probabilistic graphical model) with NN which represents
unknown interactions between variables. Show how to compute the likelihood using EM. Not read because was too
difficult.

# [A neural network approach to statistical pattern classification by semiparametric estimation of probability density functions]()
1991.05

Authors want semiparametric method because argue that previous nonparametric methods (Parzen Window or Probabilistic Neural Networks) are inadequate when we have
large training set. We still don't want parametric methods all the time when we don't know functional form of the
density function. Here author propose the mixture of component densities. The components are parametrized but not
mixture. Authors present iterative method of finding parameters of the model (mixing and mixture components) and then
method using stochastic gradient descend. The components are parametrized using non-linearities (NN squashing function).

# [Learning bias in neural networks and an approach to controlling its effect in monotonic classification]()
1993

Authors notice the bias in classification task performed by NN depends on many factors: architecture, parameters of the
learning algorithm, initial state, sequence of training points, stopping criteria for learning. Actually we cannot control
the biased boundary represented by the NN. If we imagine to decision boundaries for positive and negative examples. These
2 boundaries are composed of the efficient frontiers i.e. points the dominate or are dominated by all other points
in their region. The region between these boundaries is ambiguity region and the NN places its decision surface arbitrarily.
Authors want to fix it using monotonicity constraint. The model is composed of 2 NN that generate 2 estimates one for positive
classification and the other for negative. The result is averaged to get unbiased result. Authors demonstrate the performance
of the model on the generated data set in R^3, points separated by plane which is perfect decision boundary. The proposed
model achieves much better results than regular NN.

#[Application of the back propagation neural network algorithm with monotonicity constraints for two-group classification problems]()
1993

Again paper using monotonicity to improve classification using neural networks. Albeit it does not model the probability
of the data like our model.

#[A learning law for density estimation]()
1994

Authors use NN (with one hidden layer) to output density. 
They use output exponential unit so they approximate the family 
of exponential functions. Because they output directly density they have to normalize it. 
They maximize the log likelihood (the same is minimizing KL divergence between the true density and estimator) 
and the target density is not known  therefore it is unsupervised learning. 
The bias of the last layer is learnt so the density estimator is normalized (the learning rule is derived from the KL divergence
and the constrain for the coefficient so the density is normalized). To compute the update the computation of multidimensional
integral is required (over the entire support of the density function, therefore the support has to be compact).

# [Mixture Density Networks](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf)
1994.02

Authors notice that while for classification problems output of the model is correct posterior, in case of the regression
average conditional on the output might be very wrong if multiple outputs are possible. Authors also derive error function from different perspectives:
conventional least squares, ML, cross entropy. 
The Mixture Density Network is NN that parametrizes the mixture of Gaussians. Its outputs are parameters depending on x for mixing
probabilities and Gaussian parameters. Everything is analytically derived. Author shows derivation for empirical cost and limt
case using expectations. Having this distribution we can compute specific values like conditional mean, most probable value, variance of the
density about the conditional variance.
Paper depict 2 interesting examples of inverse mapping where forward mapping is proper function (i.e. only one value for each vector in the domain
but inverse problem having possible many outputs for each input). First is increasing sinus function. The other is kinematics of robot arm
with 2 joints where angles of joints are input and position of end of arm is output. Paper shows that it is beneficial to gave conditional 
distribution instead of only conditional mean which can be completely misleading.

# [A Neural Network Method of Density Estimation for Univariate Unimodal Data](https://link.springer.com/article/10.1007/BF01415012)
1994.02.04
They new already at this time that more hidden neurons implies less local minima. Authors **learn CDF by using NN and compute
derivative to compute pdf**. To obtain target labels for NN they computer empirical CDF from the data. They smooth it before feeding 
into NN (checking if points are concave upwards before mode and downwards after mode). **They also use monotonicity and concavity constraint** 
where unimodal cdf had to be convex before mode and concave after mode. They add new point to the training data where the mode is found
 by some additional method. They also normalize the data to fall into (0,1) to (0.2,0.8) ranges.
They use one hidden layer with known approximation of required number of hidden neurons to be (N-1)/3 where N is rank of the training data.

# [Estimating conditional probability densities for periodic variables]()
1995

Authors show method to estimate conditional density for periodic variables like direction of the wind. Authors provide 3 methods:
1. Transforming periodic variable to Euclidean space where standard methods can be used 2. Using parametric periodic kernels 
3. Using non-parametric kernel but mixture parameter parametrized. Authors test models on generated data and data for satellite.

# [Hints]()
1995

Authors add auxiliary data(hints) to the training data to guide learning process. The experiments are done on financial
FX data. Authors show how VC dimension decreases when we add hints (the VC dimension tells how much training data is enough).
The hints can be provided by creating virtual examples for example which are evaluated on additional error function (there is error function
per hint), do depending on the example (coming from training data or hint) we use different error function. There are different
ways how to combine these different error terms. 

# [Bayesian neural networks and density networks]()
1995
Learning logistic linear regression when probability over parameters is learnt. First author show that using gaussian prior
we end up using standard regularizer. Show than multiplying prior and likelihood we obtain probability over parameter space. 
Author defines probabilistic equations for posterior densities of parameters, hyper parameters (we don't have to run 
cross-validation). The model uses gaussian approximation and MC samples to compute posterior distribution.
Author notices that current NN models do not model density of the inputs. They discuss how to use MLP as density model.
The density network estimates contains hidden vector x (with prior p(x)) and has nonlinear mapping p(t|x,w) 
with priors over w, the t is data observed. The density network is similar to the second part of the autoencoder. The model can
be modified to work with partially missing input.

# [Probability Density Estimation using Artificial Neural Networks](http://www.cs.uoi.gr/~arly/papers/CPC2001.pdf)
2000.07.18

Authors propose pdf estimation using NN. They notice that it is more robust for example than mixture of Gaussians which cannot
efficiently approximate for example uniform distribution. The problem with NN estimator is that it doesn't integrate so we have to do 
it numerically. Authors also mention about the method that uses classification NN to compute pdf by learning discriminator between
data distribution and well known distribution. Using Bayes rule we can compute pdf(x). But their method is different and more
standard because it uses maximum likelihood.
The proposed model computes pdf using NN with 1 hidden layer and normalizes the probability by integrating over compact subset (assumption).
So the minimization is split into 2 parts i.e. simple derivatives of NN wrt parameters and complicated derivative of the log integral
over the whole compact space of the inputs. The integral is approximated using simple equidistant point trapezoidal rule. The points in grid
for integral approximation are chosen for each computation so optimization does not depend on them.
The compact set is created by hyperrectangle containing all training data plus some margin for pdf to fade out.
They initialize the NN with data created by using regular non parametric approximator (PZ with Gaussian).
Authors notice nice artifact, the cost function makes pdf large at training data and 0 on integration points. Because of it when
there are a lot of points in the integration grid and non integration point than pdf can spike in this place. One way to tackle this 
problem is to put integration points where data is but it only can be done efficiently in 1d. 
They test the model using several test sets generated from: mixture of uniforms and gaussians. The accuracy of approximation was
checked by plots of pdfs (gaussian mixture shows very wiggly behavior while NN approximation is very close to the generating pdf.) and
by comparing the model log likelihood to the true one.
The last example show 2-d uniform where NN again wins over mixture of Gaussians.
The method is better alternative for low dimensions aprox. upto 4. Above that the problem is numerical quadrature used to compute normalizing
integral.

# [Density estimation and random variate generation using multilayer networks](https://ieeexplore.ieee.org/abstract/document/1000120/)
2002.08.07

The same paper as "Neural Networks for Density Estimation in Financial Markets" from 1998.
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

# [RNADE The real-valued neural autoregressive density-estimator](https://arxiv.org/abs/1306.0186)
2013.06.02

The ides is based on Mixture Density Network and factorial representation of the pdf i.e. P(X)=P(X_1)P(X_1|X_2)P(X_3|X_1X_2) ...
Authors compare their model to Mixture Factor analyzers, MoG. They use MoG and MoL as the representation of the pdf in the last 
NN layer. They use various data sets for the comparison.

# [A Deep and Tractable Density Estimator](https://arxiv.org/abs/1310.1757)
2014 

Extension to the RNADE to deep models and any order of variables in autoregressive representation. This is simple multi layer
neural network with specially designed learning procedure that allows to share weights for all orderings of autoregressive
model. Authors use approximation to the exact likelihood by computing estimator which sample from the expectation. The sample contains 
only one autoregressive factor so by masking o>=d in the input and using gradient only from 

# [NICE: Non-linear independent components estimation]()
2014
Transforming data with complex dependencies into factorized distribution using transformation of density formula.
Using specially designed transformation so determinant of the Jackobian is simple to compute.
Nice comparison of this model to the VAE (entropy to determinant part, prior to the encoder part).
In experiments authors also add noise into the data and call it dequantization.
Here authors also use log-likelihood for multidimensional data so can use in my experiments.
The NICE model can easily generate samples because all transformations are easily invertible so it is enough to 
sample from factorial distribution and transform the generated code by the inverse map.
The data used in experiments in graphics so we can inpating with our model as the results (actually authors are
doing this).
Authors do inpainting using projected gradient ascent on the likelihood which can also be used for our model (iterative 
stochastic gradient procedure that maximizes log likelihood by updated missing values).

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

# [Conditional probability density estimation using artificial neural network](https://ieeexplore.ieee.org/document/7338597/)
2015.10.14

They show some method of transforming real values to vectors by bucketing. The paper is very weak.

#[MADE Masked Autoencoder for Distribution Estimation]()
2015

Using Autoencoder with each weight matrix multiplied elementwise by the mask matrixes which cause that outputs
depend on some prceding inputs with some ordering. Here they model binary data. This way the output is valid
likelihood. There are several variants like using different masks. The hidden connections also can be 
sampled during training so the model learns different ordering and over different connections. This mdoel has advantage 
over the autoregressive model because it can compute the likelihood in one pass trough the network and not like
deep RNADE which does O(D). Therefore it can have efficient GPU implementation.

#[Weight Uncertainty in Neural Networks]()
2015

Using variational approximation for posterior over NN weights. We obtain NN that does have Gaussian distribution over each weight.
Such NN can be used for example in contextual bandit problem.

# [Connectionist multivariate deMADE Masked Autoencoder for Distribution Estimationnsity-estimation and its application to speech synthesis](https://www.era.lib.ed.ac.uk/handle/1842/15868)
2016

PhD Thesis about RNADE. Good overview of the connectionist density estimation methods. Author discusses his and other
methods strong and weak points, computational cost. Overview of many autoregressive methods like LARN.
The interesting point is comparison of the deep RNADE method to the denoising autoencoder or drop out. In all cases some values
are masked but the rationale is a bit different.

This thesis is composed of 3 papers:
* RNADE The real-valued neural autoregressive density-estimator
* A Deep and Tractable Density Estimator
* Modelling acoustic feature dependencies with artificial neural networks: Trajectory-RNADE

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

#[Deep Residual Learning for Image Recognition]()
2016 



#[Density estimation using Real NVP]()
2017

The following work after NICE model. Here the same authors apply more complex transformation which preserves the
quality of being easily invertible and easy to compute determinant of the Jackobian.
Transformation from random numbers into data x. Model is composed from stacked coupling layers. Coupling layer copies one part
of the u into x and the other part is scaled and shifted using functions depending on the first part. This makes coupling
layer invertible and determinant of the Jackobian simple to compute.
Authors use trick to decrease computational and memory burden - factoring our half of the dimensions at regular intervals.
We could reuse possibly this technique.
They also use batch normalization, weight normalization and residual networks to improve propagation of the signal - we can
also try that out.
They do few data transformations I was not aware of to remove boundary effects (image data).

# [Masked autoregressive flow for density estimation]()
2017

MAF are models that use autoregressive models (MADE) for the transformation in the normalizing flow.
Authors reiterate the autoregressive model can be interpreted as normalizing flow (they show equations for that), they 
admit that it was shown originally in "Improved variational inference with Inverse Autoregressive Flow".
Masked Autoregressive Flow (MAF) is simply stacked MADEs with Gaussian conditionals to create more flexible models. 
The difference between IAD (also uses stacked MADEs) is that MAF transforms data points x_{0..i} through model and IAF processes
random vectors u. MAF computes pdf of x in one pass but sampling require D passes. IAF can generate sample and compute its
density in one pass but require D passes to compute density of provided externally data point. 
Authors show that Real NVP and NICE are special cases of MAF and IAF. The advantage of the Real NVP over MAF and IAF
is that it can compute density and sample on one forward pass.
Our MONDE using MADEs is kind of MAF but with more flexible conditional because we are not using parametrized distribution.
In experiments authors use early stopping of 30 epochs and the same UCI data sets. They use batch normalization and
regularization. Authors use also paired t-test to compare the models.
This is the original paper the used these bigger UCI data sets to assess density estimators (Power, GAS, Miniboone ...).
The authors test 3 different types of MAFs. MAF(5) with 5 autoregressive MADE layers, MAF(10) and MAF(5) MoG which is MAF(5) with 
MADE MoG (5) as base distribution (both models are trained jointly).
They use conditional density estimation on MNIST and the CIFAR-10 natural images.

# [Variational lossy autoencoder]()
2017

VAE using recurrent neural network so powerful latent representation with flexibility of the autoregressive model.
This way model excels on log-likelihood evaluations and learns interesting global representations of data.
The authors address the problem that when autoregressive model (used for p(x|z) i.e. decoding distribution) 
is used with VAE it take all burden in modeling latent code and VAE latent variables are under-utilized. This happens
because in the beginning of learning q(z|x) has little information so the training criterion shapes q(z|x) into p(z) (D_KL(q(z|x)|p(z))).
Authors make remark to bits back coding to explain this phenomenon.
Authors propose parameterizing the learnable prior as autoregressive normalizing flow. They show that it is equivalent of 
using Inverse  Autoregressive Flow for posterior but AF prior has more expressive generative model.
VLAE model can be defined as a VAE that uses AF prior and autoregressive decoder (PixelCNN).

# [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779)
2018.04.03

Generalization of the MAF and IAF normalizing flow. Here authors use NN as the transformation of the x_t and autoregressive
part (parameters depending on the x_1 ... x_{t-1}). Authors use the model in density estimation and variational inference
showing improvement over the previous models. Their approach also uses MADE to encode the parameters but the final tranformation
is monotonic NN (so it is invertible). Of course to invert it the numerical procedure have to be used but it is not necessary
for application of the model.
Nice tutorial on flows:
https://blog.evjang.com/2018/01/nf1.html
https://blog.evjang.com/2018/01/nf2.html

# [From CDF to PDF --- A Density Estimation Method for High Dimensional Data](https://arxiv.org/abs/1804.05316)
2018.04.15

Authors improve on method described in "Density estimation and random variate generation using multilayer networks". 
They improve upon this method by **making the output monotone** (so no hyperparameter tuning is necessary) increasing 
(by constraining the weight to be positive) and point to algorithm to compute efficiently high order derivative to compute pdf 
from the CDF (by using someone's else algorithm  to determine coefficients of the polynomial, we need it to compute n-th derivative of tgh which is needed
to compute **higher order derivative**, but they don't show this apprach in the experiments). 
Authors also notice that using sigmoid functions can be inefficient for approximating
strong non-smoothness like uniform distribution. They use combination of tanh and -1,x,1 function. They made also combination
parametrized to it can be fine tuned by running optimization.
As examples authors generate different mixtures of well known distributions. When using mixture with uniform which is not differentiable
using special activation function with tunable parameters helps a lot and almost ideal pdf is reached.
Currently author is working on efficient high order differentiation. Also the future work will be focused on deep nets and making it
possible to do higher order differentiation in them.

# [Introduction to Normalizing Flows](https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2018-09-Introduction-to-Normalizing-Flows/slides.pdf)
2018

Very nice tutorial on normalizing flows.

# [Normalizing Flows for Probabilistic Modeling and Inference]()
2019

Thorough overview of the normalizing flow. Authors have proof why NF can estimate any distribution meeting some requirements.
This proof uses CDFs so possibly can be extended to our models. The review of the flow discrete flows using conditioner and transformer
at each step of the flow. The second type of flows are continuous flows which describe the flow using ODE (the complexity of computing 
forward and reverse of the flow in this case is the same, which is not true for discrete flows)

#[Normalizing Flows: An Introduction and Review of Current Methods]()
2019

Review of the normalizing flows methods. Review of data sets used to validate their performance. Overview of open problems.

#[Invertible residual networks]()
2019 

#[Uncertainty Quantification in Deep Learning](https://www.inovex.de/blog/uncertainty-quantification-deep-learning/)
2019

Nice explanation of aleatory and epistemic uncertainty in NN models with simple examples.  