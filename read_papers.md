# Learning Deep Architectures for AI

Monograph about DBN. Contain research directions for the future.

“Any probability distribution can be cast as an energy-based models”
When there are no connections between hidden units then free energy and numerator of the likelihood are tractably computed.
Conditional RBM which can use generalized CD can be used to successfully model conditional distributions P (x t |x t−1 , x t−2 , x t−3 ) in sequential data of human motion [183].
Temporal RBMs have been proposed [180] to construct a distributed representation of the state. The idea is an extension of the Conditional RBM presented above
Using global optimization strategies.
Continuation method allowing to find better local optimum (aiming in finding global optimum)
Curriculum learning by first presenting simple problem to learn, increasing difficulty with time. Similar to the way human learn.
Research problems at the end of the monograph:
Can the principles discovered to train deep architectures be applied or generalized to train recurrent networks or dynamical belief networks, which learn to represent context and long-term dependencies?
Although much of this monograph has focused on deep neural net and deep graphical model architectures, the idea of exploring learning algorithms for deep architectures should be explored beyond the neural net framework. For example, it would be interesting to consider extensions of decision tree and boosting algorithms to multiple levels.

# Representation Learning: A Review and New Perspectives

Learning features is very important for machine learning.
What makes representations good.
Local non-parametric learners are not enough
Features should be distributed,invariant, disentangle factors of variation
feature invariant representation which really require defined task vs. disentangling factors where we want to keep as much of useful information in the input space as possible.
It is still not clear how to define objective to disentangle factors of variations
Good reference point for other papers about representation learning.
Representation learning is split to probabilistic graphical models and neural networks
sparse coding is popular for feature extraction
Extensions of RBMs for real values data. It is important that hidden variables encode conditional mean and conditional covariance.
Learning posterior on h i.e. P(h|x) like in RBM still require to get numerical values for representation but the features can be directly computed like in autoencoder.
Temporal coherence … can it be useful for time series ?

# A review of unsupervised feature learning and deep learning for time-series modeling
Unsupervised feature learning versus learning representations for time series data to capture temporal information (what is temporal information)
“This makes a generative model more robust for noisy inputs and a better outlier detector.” ??
“Furthermore, if the data has a temporal structure it is not recommended to treat the input data as a feature vector” ??
My Thought: Looks like there is limited knowledge in the scientific community about modeling time series financial data. For example here they consider only stock data with L1 prices. But FX data is much richer.
My Thought: The problem is the financial institutions don’t want to share the knowledge like what kind of manually extracted features they use.
“For stock prediction, the progress has stopped at classical neural networks”
“For high-dimensional temporal data such as video and music recognition, the convolutional version of RBM have been successful.” - My Thought: maybe try to use it to model FX data.
My Thought: Currently we model time series using equidistant measurements (by discretizing original measurement). Do we loose information with this approach. Should we use the original timestamps of all measurements and what models should be used then.
“Another possible future direction is to develop models that change their internal architecture during learning or use model averaging in order to capture both short and long-term time dependencies” - had similar idea in my research ideas.
“Further research in this area is needed to develop algorithms for time-series modeling that learn even better features and are easier and faster to train. Therefore, there is a need to focus less on the pre-processing pipeline for a specific time-series problem and focus more on learning better feature representations for a general-purpose algorithm for structured data, regardless of the application.”
“The second characteristics of time-series data is that it is not certain that there are enough information available to understand the process.”
“In financial data when observing a single stock, which only measures a small aspect of a complex system, there is most likely not enough information in order to predict the future [30].”
The y(t) we predict depends on x(t) i.e. not only on x but also on t. To predict we have to provide more data input from the past or model has to remember the history.

# On the Need for Time Series Data Mining Benchmarks: A Survey and Empirical Demonstration
Comprehensive survey of recent work on time series data mining 
Most paper on time series data do not properly validate its results
Methods compared to very few rival methods (median 1)
Small number of datasets used to check the method
“Implementation bias is the conscious or unconscious disparity in the quality of implementation of a proposed approach, vs. the quality of implementation of the competing approaches.“
“Data bias is the conscious or unconscious use of a particular set of testing data to confirm a desired finding.”
When comparing to rival methods implementation details of both methods can significantly affect the comparison. 
Normalizing time series data is very important.
“However the unique structure of time series means that most classic machine learning algorithms to not work well for time series. In particular the high dimensionality, very high feature correlation, and the (typically) large amounts noise that characterize time series data have been viewed as an interesting research challenge.”
My Thought: When features are built from time series then use similarity measure on the new space and create for example dendrogram of several time series with similar qualitative patterns.
My Thought: Using Piecewise Linear Representation to align different time series. (or try any other segmentation techniques)

# Machine Learning for Sequential Data: A Review
normal classification supervised learning does not take sequential dependence into account
sequential supervised learning (h(x_vec)=y_vec) versus time series prediction ( y_{t+1} = h(y_t))
My Thought: Using cost matrix for time series prediction or for event detection
feature selection problem: wrapper approach, using penalty on meaningless features,  measure of feature relevance,  first fit a simple model and then analyze the fitted model to identify the relevant features
Hidden Markov Model, Maximum Entropy Markov Model, Input/Output Hidden Markov Model

# 10 CHALLENGING PROBLEMS IN DATA MINING RESEARCH
noise in time series data is still problem for machine learning
non-i.i.d. data. My Thought:  Is L2 data non-i.i.d ?


# An Introduction to Hidden Markov Models
Tutorial on HMM.
within short time some signals can be effectively modeled by a simple linear time-invariant system
Three problems: estimate probability of sequence given the model, find the most probable sequence of states, find parameters of the model.
HMM using dynamic programming to get probability of the sequence (forward backward procedure)
Viterbi algorithm used to get most probable sequence.
HMM are used to discover higher level structures in the signal (features)
	
# Composable, distributed-state models for high-dimensional time series (PhD)
Work about DL models for multivariate time series.
Possibly contains good references to read.
Compares expressive power of models with only visible variable (Moving Average, Vector Autoregressive Models, Markov Model) to models with also hidden variables (HMM)
Distributed state models like FHMM and PoHMM
state space models with real valued state vector (like Linear Dynamic System, Kalman Smoothing)
Sigma Point Kalman Filter as extension of EKF
We now introduce sequential Monte Carlo methods (more commonly known as particle filtering) as another variant for approximate inference in continuous state-space systems. 
Neural Networks
Representing time series in FFNN by transforming to spatial representation is not good.
Elman Neural Network as extension of NN for time series, used truncated BPTT
Time Delay Neural Networks uses full BPTT
Research Idea: “Unfortunately a major challenge of training TDNNs is that the gradients eventually disappear as they are propagated back in time. This is simply a consequence of multiplying partial derivatives. Training is also computationally expensive, though the recent popularity and availability of parallel computing environments may rekindle interest in such models.”
Temporal Boltzmann Machines
Conditional restricted boltzmann machines and higher level models: Conditional Deep Belief Networks. There are also Temporal RBMs which contain temporal links between hidden units.
Sparse representations are often more easy to interpret, and also more robust to noise. 
Stopped reading because need more knowledge to understand the material.

# An Introduction to Conditional Random Fields

“CRFs are essentially a way of combining the advantages of discriminative classification and graphical modeling, combining the ability to compactly model multivariate outputs y with the ability to leverage a large number of input features x for prediction”
“we discuss relationships between CRFs and other families of models, including other structured prediction methods, neural networks, and maximum entropy Markov models”
“The main conceptual difference between discriminative and generative models is that a conditional distribution p(y|x) does not include a model of p(x), which is not needed for classification anyway.” - My Thought: does this sentence contradict the pretraining for deep models?
When we have a lot of training data and we are not afraid of overfitting we should use discriminative model: “it can be seen that the conditional approach has more freedom to fit the data… On the other hand, the added degree of freedom brings about an increased risk of overfitting the training data, and generalizing worse on unseen data.”
Research Idea: “whereas unsupervised learning in discriminative models is less natural and is still an active area of research.” … Unsupervised learning in discriminative models for time series??

# Scaling Learning Algorithms towards AI
Paper showing why deep learning makes sense.
“Although the human brain is sometimes cited as an existence proof of a general-purpose learning algorithm, appearances can be deceiving: the so-called no-free-lunch theorems [Wolpert, 1996], as well as Vapnik’s necessary and sufficient conditions for consistency [Vapnik, 1998, see], clearly show that there is no such thing as a completely general learning algorithm. All practical learning algorithms are associated with some sort of explicit or implicit prior that favors some functions over others.”
“Non-convex loss functions may be an unavoidable property of learning complex functions from weak prior knowledge.”
In practice, prior knowledge can be embedded in a learning model by specifying three essential components:
1. The representation of the data: pre-processing, feature extractions, etc. 
2. The architecture of the machine: the family of functions that the machine can implement and its parameterization.
3. The loss function and regularizer: how different functions in the family are rated, given a set of training samples, and which functions are preferred in the absence of training samples (prior or regularizer).
“shallow circuits are much less expressive than deep ones.”
Kernel machines need a lot of kernels and data to approximate function with large variability (which changes signs often)
Unsupervised training layer by layer is better than supervised layer by layer
“For example, our results suggest that combining a deep architecture with a kernel machine that takes the higher-level learned representation as input can be quite powerful.”

# Why Does Unsupervised Pre-training Help Deep Learning?
regularization or optimization effect when using pretraining.
how to check we reached local minimum “ It is difficult to guarantee that these are indeed local minima but all tests performed (visual inspection of trajectories in function space, estimation of second derivatives in the directions of all the estimated eigenvectors of the Jacobian) are consistent with that interpretation.”
My Thought: how to check that different local minima are really different?
Test error variance with pretraining is smaller than without pretraining. (regularization explanation), variance reduction technique.
With all layers small the benefit from pre-training disappears (which points to regularization effect)
My Thought: Is pretraining and then supervised learning isn’t like learning with increasing complexity of loss function?
Pre-training also helps when the data is “infinite”
Measuring test error in online setting by learning and testing on not seen data (in infinite stream of data setting)
SDAE seems to do better than RBM for pretraining.
In case of stochastic on line training the order of presentation of the examples makes a difference, especially the examples in the beginning and less examples presented later. The same can be noticed in neuroscience/psychology: “This would behave on the outside like the “critical period” phenomena observed in neuroscience and psychology (Bornstein,1987)”
My Thought: In case of time series we should cater for early examples influencing learning the most (curriculum learning)?

# Feature-based Classification of Time-series Data
Using NN for time series classification
Using features as input instead of raw data: “Since features carry information for the time series which is not based on individual time points, they are less sensitive to noise.”
Classification of artificially generated time series (4 equations with random component)
Research Idea: Using time series classification to select model depending on the state of the market.
Why using features is better than raw data:
It is sensitive to the amount of noise in the time series. Since each input contributes to the classification task, noise can affect the accuracy of the method more severely.
It is sensitive to the length of the time series. When the time series becomes large, it is impractical to have a neural network with such a large number of inputs.
It requires that all chart patterns are of the same length (due to the constant number of input neurons). Using features instead, patterns of di erent lengths could be used, as long as the same number of features is extracted.
Research Idea: I can use similar features in my comparison i.e. statistical features (first and second order features) versus expert features vs learned features.
The model using features achieves better accuracy than model using raw data when we increase the noise.
Also when length of the time series increases the accuracy of raw data model is much worse than feature based model.
Feature based model takes also much less time to train (especially for models using long time series input)

# The behavior of stock-market prices
random walk against chartists
My Thought: How to differentiate between importance of the model from the statistical point of view and importance viewed by the trader (paper mentions this idea)
intrinsic value is not at odds with random walk theory
“In sum, this discussion is sufficient to show that the stock market may conform to the independence assumption of the random walk model even though the processes generating noise and new information are themselves dependent”
dependence in information and dependence of valuing intrinsic value by different agents (dependence in noise generating process).
The theory of random walks in stock prices is based on two hypotheses(my work can support or reject this hypothesis):
 successive price changes in an individual security are independent
the price changes conform to some probability distribution
sophisticated traders dampen the dependence in the noise generating process (agent following other agents who are deemed opinion generating)
“In sum, this discussion is sufficient to show that the stock market may conform to the independence assumption of the random walk model even though the processes generating noise and new information are themselves dependent. We turn now to a brief discussion of some of the implications of independence.”
Paretian distributions are good model of distribution of price changes
For example, if the variances of distributions of price changes behave as if they are infinite, many common statistical tools which are based on the assumption of a finite variance either will not work or may give very misleading answers. 
“Similarly. under the Gaussian hypothesis for any given stock an observation more than five standard deviations from the mean should be observed about once every 7. 000 years . In fact such observations seem to occur about once every three to four years.” - check for Swiss peg removal
Research Idea: Try to fit the FX returns to the Paretian distribution and check the value of the characteristic exponent a. When it is 2 then the distribution is Gaussian and if <2 then the tails are fatter. Paper contains different ways of assessing it.
Serial correlation and run test to discover data dependence.
My Thought: “Write chapter/experiment to check if HF FX data meets random walk assumptions”
The conclusion is that the past prices cannot be used to increase investors’ returns.
“In general, when dealing with stable Paretian distributions with characteristic exponents less than 2, the researcher should avoid the concept of variance both in his empirical work and in any economic models he may construct.” - should we instead use mean absolute deviation which is more stable for Paretian distribution with characteristic exponent less than 2?

# Planning and Control (Excerpt from 􏰚”Planning and Control􏰛”)
Introduction to dynamical systems
Separation property means that observation problem and input-regulation problem can be solved separately. This is the case of the linear dynamical system corrupted by Gaussian noise and subject to quadratic performance criteria. Separability guarantees that􏰊 when the observer and regulator are coupled together􏰊 the resulting controller will be optimal with regard to the stated criteria􏰃
Introduction to Learning Dynamical Systems
http://cs.brown.edu/research/ai/dynamics/tutorial/Documents/DynamicalSystems.html

# Learning dynamic bayesian networks
Bayesian Learning of the model and then the parameters of the model:
	
	
And then Bayesian prediction is:
	
Perspectives on system identification
For the curve fitting problem relation between fit for the train and validation data:
	
The first expression is Akaike’s Final Prediction Error (FPE), the second one is Akaike’s Information Criterion (AIC) when applied to the Gaussian case (Akaike, 1974), and the third one is the generalized cross-validation (GCV) criterion (Craven & Wahba, 1979). Here d = dim u serves as the complexity measure of the model set.
Also other more generic formulas for validation error bounds (VC)
Information content in the data (Fisher information matrix), Crame´r–Rao inequality
	where hat is unbiased estimator.
building mathematical models of (dynamic) systems done in many scientific communities:
Statistics and statistical learning theory
machine learning
Manifold learning - “It can be described as a way to determine a coordinate system in the manifold that inherits neighborhoods and closeness properties of the original data.”
Econometrics
Statistical process control and chemometrics
Data mining
Artificial neural networks
Fitting ODE-coefficients, and special applications
System identification
Some open areas in system identification
Issues in identification of nonlinear models, “Identification of nonlinear models is probably the most active area in system identification today”
Convexification (Research Idea: Can it be applied for deep learning?); globally identifiable model structure can be rearranged (using Ritt’s algorithm) to a linear regression
Model reduction 
Nonlinear systems–nonlinear models
Model types with respect to model reference to the real object, from white to black (gray box model - mixing physical insights with information from measured data)


# A Tutorial on Fisher Information

“The (unit) Fisher information is a measure for the 18 amount of information that is expected within the prototypical trial X about the parameter 19 of interest θ. It is defined as the variance of the so-called score function, i.e., the derivative 20 of the log-likelihood function with respect to the parameter”

System identification of nonlinear state-space models
“Unlike gradient based search, which is applicable to maximisation of any differentiable cost function, EM methods are only applicable to maximisation of likelihood functions.“
Research Idea: Compare model from this paper and the one in “Learning deep dynamical models from image pixels”

# Maximum Likelihood from Incomplete Data via the EM Algorithm
Good intro to EM with simple examples

# An Overview of Sequential Monte Carlo Methods for Parameter Estimation in General State-Space Models
Sequential Monte Carlo (Particle filters) as alternative to Extended Kalman Filtering for nonlinear, non-Gaussian dynamical models

# Reducing the Dimensionality of Data with Neural Networks

RBM learning for binary data
Autoencoder: Pretraining, Unrolling, Fine-tuning
The code layer is linear and other logistic
Research Idea: When extracted features size 2 or 3 show them on the plot adding color for points when return increased or decreased to see if there are noticeable clusters like with digits data.
Looks like for autoencoder with increase and then decrease the size of the layers like: 625-2000-1000-500-30 … why? - answered in the supporting material - “We used more feature detectors than pixels because a real-valued pixel intensity contains more information than a binary feature activation”
“Pretraining helps generalization because it ensures that most of the information in the weights comes from modeling the images. The very limited information in the labels is used only to slightly adjust the weights found by pretraining.”
Pretraining and fine tuning can be applied to very large data sets because both scale linearly in time and space with the number of training cases
Research Idea: Use code http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html to check of time series data.
From experiments it looks like pretraining RBMs longer do not help because it is enough if initial weights are set up in the good region before fine tuning.
Compares deep and shallow autoencoder with the same number of parameters and shows that deep is better. Research Idea: repeat the same experiment.
Research Idea: Normalize the data so the returns are -1 ,0 or 1 and see how classification would work.
“To further reduce noise in the learning signal, the binary states of already learned feature detectors (or pixels) in the “data” layer are replaced by their real-valued probabilities of activation when learning the next layer of feature detectors, but the new feature detectors alway have stochastic binary states to limit the amount of information they can convey.” i.e. when doing CD then hidden layer will have binary values set but when it is used as data to the next RBM it uses real valued probabilities.
“the aim of the pretraining algorithm is not to accurately reconstruct each training image but to make the distribution of the confabulations be the same as the distribution of the images.”

# A Fast Learning Algorithm for Deep Belief Nets
“The generative model makes it easy to interpret the distributed representations in the deep hidden layers.”
The RBM is equivalent to infinitely stacked Sigmoid Belief Net.
Labels are attached as softmax at the top auto associative memory (RBM). Then when classifying computed probability of posterior.

# Connectionist learning of belief networks
Belief networks can be learnt using expert knowledge and from the data.
Research Idea: When learning use the models first on some simple problem like mixture model.
Gradient Ascent algorithm to learn Boltzmann Machine (gradient of likelihood of data), contains negative and positive phase (negative because of Z).
Gradient Ascent algorithm to learn Sigmoid Belief Networks and noisy or Belief Networks (not containing negative phase). “Intuitively, only a single phase is needed for learning in a sigmoid belief network because normalization of the probability distribution over state vectors is accomplished locally at each unit via the sigmoid function, rather than globally via the hard-to-compute normalization constant, Z. ”
There is no difference in representational power between 0/1 and -1/1 networks.
Hidden units in Boltzmann Machines or SBN allow to represent mixtures over visible variables:

We can model different characteristics in the data using architecture of choice:

My Thought: As in this text before experiment we can list the questions which should be answered by the experiment.
My Thought: Describe experiments quite thoroughly.
-Data Loglihood/number of cases is the entropy, and if log2 is used then performance used in bits (smaller better)
The SBN tolerated much bigger learning rates than Boltzmann machine.
My Thought: The learning with SBN achieves better results than BM. Why then we use BM in pretraining of deep network and not SBN.

Training Products of Experts by Minimizing Contrastive Divergence (not finished)
Good explanation of CD
PoE can be better than MoE for multidimensional distributions 
Research Idea: Can we model multidimensional Time Series data using PoE
Research Idea: Test time series on RBMs


Extracting and Composing Robust Features with Denoising Autoencoders
what criteria a good intermediate representation should satisfy: 
retain a certain amount of “information” about its input
being constrained to a given form
sparsity
robustness to partial destruction of the input
When using small additive noise then it works as regularizer (smoother) but in this case information is destroyed (big noise) 
Research Idea: Using filtering approach create true version of the time series (FX) and then using raw signals as noisy input try to build representation model of it like in denoising autoencoder.
The denoising autoencoder can be viewed as the way to define and learn a manifold.
The learning procedure can also be derived from a generative model perspective
It also maximised the lower bound on I(X,Y) - mutual information where X is the input and Y is learned representation.
Displaying filters (weights) after learning sDAE we see that when corruption is increased then more global features are learned.
My Thought: For time series also display filters to see if anything meaningful can be seen.

# Auto-Association by Multilayer Perceptrons and Singular Value Decomposition
Auto-association (autoencoder) is the idea known since at least 1986.


# Learning deep dynamical models from image pixels
My Thought: When we use VWAPs and others it is like low dimensional representation of our data. Possibly model like sDAE could build better representations.

# Sparse deep belief net model for visual area V2
Sparse RBM algorithm (normal RBS is not sparse) (Research Idea: Can use it instead of autoencoders)
When learning the model with optimization problem composed of log likelihood and regularizer they use separately contrastive divergence for log likelihood and gradient descent for regularizer.

# Notes on Contrastive Divergence
Explanation for CD

# On Contrastive Divergence Learning
Use CD to get biased ML solution and then use slow ML learning to fine tune.
The distribution near the boundary of the simplex are more difficult to model. Research Idea: Show the location of the distribution I model.

# IMPORTANCE OF INPUT DATA NORMALIZATION FOR THE APPLICATION OF NEURAL NETWORKS TO COMPLEX INDUSTRIAL PROBLEMS
 
 “We have found that input data normalization with certain criteria, prior to training process, is crucial to obtain good results, as well as to fasten significantly the calculations.”
different units, order of magnitude of absolute values of the inputs => “completely necessary to perform some kind of input data pre-treatment. ”
RESEARCH IDEA: Show statistics for different values like min, max, mean, median.
Different ways of transforming the data, not only using normalization. We can transform the data to keep the relative magnitude difference between variables or to remove this difference.
“In both cases it can be seen that error gradually decreases as normalization reduces the differences between the variation range of the different variables.”

# Application of Neural Networks to Signal Prediction in Nuclear Power Plant
“In the operation of a nuclear power plant, the plant operators should always observe to what condition the plant is going. However, there are so many plant parameters that the operator cannot watch all of the signal trends. From this point of view, the need to develop computerized signal prediction systems becomes significant.“ - the same rationale applies here.
Smoothing, filtering and prediction as 3 basic types of signal estimation.
RESEARCH IDEA: Test the idea with stochastically adjusting learning rate and momentum factor. Does it help to escape local minima.

# Learning Multilevel Distributed Representations for High-Dimensional Sequences
Model with componential hidden state =>  it has an exponentially large state space.
“Models that use latent variables to propagate information through time”

# An Introduction to Variational Methods for Graphical Models
Representing neural network as sigmoid belief net. The inference is intractable because during moralization all the parents become connected.
RESEARCH IDEA: Training neural networks as probabilistic methods using approximate methods like variational methods.
Variational method can be used to aproximate posterior and intractable energy term and enable tractable inference and learning.
Variational methods transform local probabilities so the inference becomes tractable and possible we can use exact methods.

# Nonlinear Multilayered Sequence Models
When building representations for temporal data it is important that every feature vector for every arriving data point uses information from the previous data points and their representations.
When graphical models with hidden features are fitted to the data their hidden states act as useful features. There is no theory explaining this fact.
TRBM using bias functions that depend on the previous hidden and visible variables.
Windowed Restricted Boltzmann Machine using local approximations. They use only undirected connections and also add intra time visible variable connections. There are no hidden to hidden connections between time steps
TRBM and WRBM are poor in representing long sequences.
Multigrid Multilayered WRBM modification of MRBM halving hidden number of variables in each deeper layer so they can look at longer subsequence. Should be bettered at modeling longer sequences.
Recursive MM-WRBM the top layers have fixed parameters and the depth is expended depending on the length of the sequence.

# The wake-sleep algorithm for unsupervised neural networks

Nature, Deep learning
Deep means 5-20 layers.
ReLU currently most popular non-linear function (half wave rectifier).
ReLU allow training deep nets without using unsupervised pretraining.
Initial conditions are not a problem for deep network, “ the system nearly always reaches solutions of very similar quality.”
“ For smaller data sets, unsupervised pre-training helps to prevent overfitting”
RESEARCH IDEA: Using transfer learning from prices to hedging. Built system that would build features from the data when we have a lot of labels like price prediction and use those features for hedging with addition of reinforcement learning.
RESEARCH IDEA: CovNets process data that come in the form of multiple arrays so it should be perfect for multi level market data.
RESEARCH IDEA: Can deep network rediscover technical analysis.
First example of 1-D ConvNet is Time Delay Neural Network.
RESEARCH IDEA: Like in image to text where ConvNet is used to create representation that is fed into RNN to create a sentence. We could use this idea for hedging where ConvNet analyzes prices and RNN is used to generate orders.
Distributed representations excel when “underlying data-generating distribution having an appropriate componential structure”. Can we check it for our data.
“Neural Turing machines can be taught ‘algorithms’.”
“Unsupervised learning had a catalytic effect in reviving interest in deep learning, but has since been overshadowed by the successes of purely supervised learning. Although we have not focused on it in this Review, we expect unsupervised learning to become far more important in the longer term.”
RESEARCH IDEA: Compare predictive capability by learning deep model in purely supervised way and purely unsupervised way.

# Deep Learning in Neural Networks: An Overview
credit assignment as important part of learning
review with a lot of useful references.
uses a bit different notation that I am used to - events.
UL is normally used to encode raw incoming data such as video in format that is more convenient for subsequent goal-directed learning.
Group Method of Data Handling networks that incrementally grow when learning, number of layers and number of units in the layer can be learnt
Neocognitron as the first NN that introduced convolutional NN.
Chapter 5.6.1 Ideas for Dealing with Long Time Lags and Deep CAPs showing possible application for temporal data
Various improvements on BP method described.
Addressing bias/variance dilemma via strong prior assumptions like weight decay
Check 5.6.4 Potential Benefits of UL for SL as good references for representational learning
UL can create factorial code (a code with statistically independent components) so we can use for example naive Bayes classifier on it.
 LSTM could also learn DL tasks without local sequence predictability
RNNs can also be used for meta learning (learning how to learn)
2007, hierarchical stacks of LSTM RNNs were introduced
RESEARCH IDEA: Have to test LTSM RNN because they work very well on sequences.
Currently Successful Techniques: LSTM RNNs trained by CTC and GPU-MPCNNs (both are SL)
The problem of RL is as hard as any problem of computer science, since any task with a computable description can be formulated in the RL framework
RNNs can produce compact state features as a function of the entire history seen so far. (functional approximation of huge state space)
Future: 
learn to actively perceive patterns (saccades)
take into account that it costs energy to activate neurons, Brains seem to minimize such computational costs
more distant future may belong to general purpose learning algorithms that improve themselves in provably optimal ways (Gödel machine)

# The Curse of Dimensionality for Local Kernel Machines (not finished, too difficult)
“The curse of dimensionality has been coined by Bellman (Bellman, 1961) in the context of control problems but it has been used rightfully to describe the poor generalization performance of local non-parametric estimators as the dimensionality increases.”
Gaussian kernel machines need at least k examples to learn a function that has 2k zero crossings along some line
Using KM when test point is far from training data the output/prediction goes to some constant (depending on the kernel)

# Neural Networks and the Bias/Variance Dilemma
very good intro to bias variance dilemma
In this 1992 article authors assume that because of high variance (bias variance dilemma) of FNN they do not allow to model the problem with small estimation error.
They show that because ML is about extrapolating (high dimension of input space) the completely nonparametric methods are too inaccurate and we need to introduce prior biases. (data representation is one of the possibilities to introduce bias)
Machine Learning can be also read as Function Learning
FNN are parametric estimators when architecture is fixed but varying the size of the network with the data size given nonparametric estimator. By increasing the size of the network with the size of the data we decrease the bias. Such estimator is consistent i.e. its mean square error (estimator to regression) goes to 0. It shows artificial boundary of nonparametric vs parametric estimator where non parametric estimator is sequence of parametric estimators indexed by data size.
Example to compute the (integrated) bias and variance of the model using MC even when regression is not know.

# On the Number of Linear Regions of Deep Neural Networks
Describes how deep NN (using rectifiers and maxouts) maps its input to exponentially many linear regions (with depth)
Each layer is a map that folds its domain so some linear regions are identified it its output (identified means that 2 regions are mapped to the same region). Then the next later can compute on this folded region ergo running the same computation on different input regions.
“The number of linear regions of the functions that can be computed by a given model is a measure of the model’s flexibility.”
“ A linear region of a piecewise linear function F : R^n_0 → R^m is a maximal connected subset of the input-space R^n_0 , on which F is linear.”
Described method of visualising weights of NN

# Learning Long-Term Dependencies with Gradient Descent is Difficult 
Difficult because about attractors 
They run experiment when sequence of length T has to be classified and the sequence class depends only on L << T first values.The rest of the sequence can also be important because it can erase information stored by the machine.
One simple neuron with self loop can latch one bit of information. First the learned sequene moves the state of the neurons to one of two attractors and then the input with small value cannot change its output. 
The system which can reliably latch information causes gradients to vanish ergo using gradient based methods is not efficient.
they use pseudo newton method to compute learning rate (inverse of the curvature) and use diagonal approximation of the Hessian.

# Masterclass on Deep Learning 2013 (Bengio)
Denoising Autoencoder used as MCMC that learnt transition operator. First corrupt the data then denoise then corrupt and denoise …
RESEARCH IDEA: Predicting using HFT data (not equally spaced and different parts coming at different times). Treat values not updated at time step as missing values. Does the model remember the previous values that are now missing ?
Dependency networks as a way to discover causal relationships  (DN as Generative Stochastic Network)?

# Pattern Recognition and Machine Learning,  Bishop Book 
Using Bayesian Variational Method we do not have to worry about overfitting and no cross validation is necessary. Apply to Deep Belief Network.

# On the saddle point problem for non-convex optimization
“However, the Newton method does not treat saddle points appropriately; as argued below, saddle-points instead become attractive under the Newton dynamics.” Because saddle points proliferation in multidimensional problems the newton methods are not appropriate. 
Hessian Free Optimisation was shown to be the variant of the natural gradient descent.
When using Newton method when rescaling gradient with negative eigenvalue we change direction of the gradient which now points into wrong direction moving to saddle point and not away from it.
Using absolute value of eigenvalues when scaling gradient so it is not attracted to saddle points.
We can check that we converged to saddle point by checking the norm of the largest negative eigenvalue.

# The Loss Surfaces of Multilayer Networks 
Approximating the fully connected FFNN with spin glass model and using Hamiltonian energy (very naïve assumption about independence of variables)
For small models it is very likely to have poor local minima at high loss function
For large models most of the critical points above critical energy are saddle points which is handled well by SGD
For large models the local minima which are not likely to be found for large loss but only close to the global minimum (they are layered)
Actually it looks that it is not worth to look for global minimum because model tends to overfit and most local minima are very close to global minimum
Additional theoretical work to be done: remove independence assumption

# A Connection Between Score Matching and Denoising Autoencoders
“This note uncovers an unsuspected link between the score matching technique for learning the parameters of unnormalized density models over continuous-valued data, and the training of denoising autoencoders” Score matching is approximate maximum likelihood technique for learning parameters of unnormalized density models over continuous variables.
“Denoising autoencoders (DAE) were proposed by Vincent et al. (2008) as a simple and competitive alternative to the contrastive-divergence trained restricted Boltzmann Machines (RBM) used by Hinton, Osindero, and Teh (2006) for pretraining deep networks”
In DAE for the continuous data the mapping into hidden layer can be nonlinear of the affine transformation and decoding can be only affine transformation using the same W. And we minimize squared reconstruction loss.
Score matching is minimize expected (wrt to true distribution) distance between score of the model and score of the true distribution (where score is different that statistical score, here score  is grad log pdf wrt data where is statistics score is wrt to model parameters). Defining score in this way allows to get rid of normalizing term.
They show that DAE is equivalent to score matching to the score of the Parzen estimator.


# Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient
Learning RBM parameters by maximizing data log likelihood using gradient descent is difficult because have to compute expectations wrt to the model (because of the partition function). This part gives negative gradient which normally has to be approximated like in CD-1. Positive phase increases energy for the data and negative phase decreases energy for the data generated from the model.
In Persistent CD we remember the previous state of the chain (hidden variables) and during next MC run we start from the point we left. Assuming that mixing rate is much larger then learning rate the chain will be able to catch up with changes to the model because of  the parameter changes.
PCD is much better than CD1 in terms of the test data log likelihood and classification (extracted features are better for discriminative task). 
PCD superiority is also demonstrated when generating data from the model. (MNIST digits)


# Deep Learning Lecture by Nando de Freitas
When to apply machine learning:
Human expertise is absent
Humans are unable to explain their expertise
Solutions changes with time
Solutions need to be adapted to particular cases
The problem size is too vast for our limited reasoning capabilities.
Challenge: one shot learning. Learn from one example, multitask learning, scaling, energy efficiency (brain consumes 30W),ability to generate data, architecture for AI.
MLE is equivalent of minimizing KL divergence (information theory) between model and true distribution  and this can be seen as contrast between true model of the data and our model (the same as in CD algorithm)
Using variational autoencoder we replace integration needed in bayesian approach with differentiation which currently scales better.
Uses a lot of Torch examples.
Nice explanation of backprop using modular backpropagation.
Applying convolutions like in world/sentence/document embedding. Like for sentence we can map level of the book window. Then map all the levels to book embedding. Then map multiple view to market embedding and then build classifier/regressor on it.
“From Machine Learning to Machine Reasoning” - learn network when we have a lot of labels (regression) and reuse features for other tasks like when clients place orders.
Max margin formulations (shuffling energy to the data and removing from the hallucinations). Hinge loss of unconstrained formulation.
Memory networks = convnets + hinge loss
Why generating data ? Alex Graves
to improve generalisation
create synthetic training data
piratical task like speech synthesis
simulate situations
understand the data
By looking what model generate you can check what model learnt.
Using prediction we can generate the data. It is the closest that computer can get close to dreaming.
RNN network generating mixture of gaussians.
Alex Graves RNN that generate handwritten text is generic. Can be applied to any time series.
Sequential process with attention (attention becomes more popular)
It looks that reinforcement learning can also be used for prediction because it can learn where to look (actions) at the data (when it is highly dimensional). We learn policy (RNN or any other neural network) which outputs actions which direct what data to get. The maximize the expected (wrt policy) reward which is normally expected value of sum of rewards. 

# Greedy Layer-Wise Training of Deep Networks
Extending DBN to continuous inputs
“we discuss a problem that occurs with the layer-wise greedy unsupervised procedure when the input distribution is not revealing enough of the conditional distribution of the target variable given the input variable”
When modeling inputs as Gaussians we add quadratic terms to energy function with precision as coefficients of visible variables squared.
They do experiment by modeling returns on financial time series.
“The results also suggest that unsupervised greedy layer-wise pre-training can perform significantly better than purely supervised greedy layer-wise pre-training.”
Experiment with decreasing number of nodes in the upper layer causes the training error to increase which can point to the fact that without pretraining we learn upper layers and lower layers are random transformations.
They consider the idea of training together all layers instead of stacking them one by one. We simply use current representations when training higher layers.
Dealing with uncooperative distributions:
x~p(x) and y = f(x) + noise (for example p is gaussian and f is sinus). No relation between x and f so unsupervised pre training can be fruitless.
In such cases it is beneficial to mix unsupervised pretraining with supervised training. (partially supervised greedy training)
They show that this approach show good results on predicting the price of cotton.

# Semi-Supervised Learning with Deep Generative Models (not read)
Paper to read because it can be related to “Auto-Encoding Variational Bayes”
		
# Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition
HMM - DNN model which combines neural network and HMM. Here NN provides observation probabilities.
# Unsupervised Neural Hidden Markov Models
Performing inference in HMM using NN. Replaces transition and emission matrices with output of the NN. The posteriors are still computed using Baum-Welch algorithm and they use 2 stage EM algorithm. The posteriors are used to weight gradients of the emission and transition which can be computed using NN. It would be interesting to compare it to our model.
Using CNN for observation distribution and RNN (LTSM) for transition model (effectively replacing 1 step Markov with dependence on all previous data) … but does it mean that data have to be fed sequentially and we cannot do stochastic optimization like we do.
Lessons learnt by authors could be important to our previous work because they found that when using NN in unsupervised model it is more important to use dropout and proper weight initialization compared to supervised models.
# A survey of hybrid ANN/HMM models for automatic speech recognition
Such models were developed already in 80s and beginning of 90s. Authors described different combinations if HMM and ANN and different ways of training. 
Separate (HMM and ANN params are learned separately) and global schemes are described (all parameters are learned together). 
Are hybrid systems worth considering a research topic or obsolete?
# A Connectionist Approach to Speech Recognition
Integration prior knowledge with learning: constraints on architecture, constraints that guide training, prior knowledge used to initialize free parameters. 
Different interesting ideas I haven’t heard before like: adding special outputs, smoothing the inputs to help with regularization. 
Partitioning of large system into smaller networks is described and training it globally and each part separately. 
Different architectures for processing sequences, more interesing: TDNN, RNN

# Dynamic Routing Between Capsules
Capsules are routed using Hebbian rules i.e. the connection between capsules on the consecutive layers are enforced if their outputs agree. Capsules are applied in the convolutional way. Don’t fully grasp the architecture so have to revisit it later.
# Transforming Auto-encoders
Network of capsules that learns to transform images. Capsule has recognition and generative units. It has also boolean gate to recognize if entity is on the picture. When applied in CNN way there is no weight sharing?
# Matrix capsules with EM routing
Reach source of references on how to improve CNNs by adding other types of invariances. Only read beginning.
# Adding Gradient Noise Improves Learning for Very Deep Networks
This uses annealed noise added to gradient. Maybe it can be compared to QA. Added gradient noise is similar to simulated annealing. 
Adding noise only helps with very deep architectures h > 5. For simple FFNN It looks that using good init and grad clipping achieves the same results as the same with 
grad noise. Grad noise helps when only simple init. But it is still useful when network is complex and we are not sure about proper init. But this is not true for more complex architectures where finding perfect init values is much harder.
# Quantum machine learning : what quantum computing means to datamining / Peter Wittek
Started but in the end was too hard to follow. Maybe later.

# Learning to Execute
LSTM that learns what is the output of the short program. The authors also test the LSTM model on copy task and notice that improvement can come from few heuristics 
like inversing the input or doubling it. Authors also notice that learning is only possible when applying their new curriculum learning (mixing simple curriculum  
and some random that can select also difficult examples like in Reinforcement learning neural Turing machines). It looks that combined curriculum  learning is mandatory 
and naive one can be worse the data distribution one.
Authors generate programs automatically that meet some criteria  regarding the difficulty. Overall this article seems to be weaker model than for example NTM.

# Mastering the game of Go with deep neural networks and tree search
Use value, policy networks together with Monte Carlo Tree Search and other techniques to achieve the best ever AI go player. First policy network is trained supervised on database of games. It is seed to policy network which is learnt using RL with self play. On these self play games value network is trained. 
# http://karpathy.github.io/2016/05/31/rl/
Karpaty also thinks that Policy gradient is much better than DQN. Policy gradient is sum A_i P(y_i|x_i) where y_i is sampled action, x_i is input and A_i is 
advantage i.e. the reward at the end. So we discourage all moves which in the end made us lost and encourage all movements that led to win. So here RL
is like supervised learning. When discounting all the rewards we can standadize them so we encourage half of the actions and discourage half of the actions.
This decreases variance of the policy gradient estimator. Policy gradient is score function gradient estimation. grad_x expectation_p(x) f(x) = 
expectation_p(x) ( f(x) * grad_x log p(x) )
# Deep Recurrent Q-Learning for Partially Observable MDPs
Using RNN on top of the DQN (CNN for modeling Q values). The paper shows experiments where RNN model helps with partial observability of the environment. They
do it but randomly clearing screens given to the model. In DQN paper authors had to give few previous frames from the video game to capute dynamics of the system,
RNN can model it implicitly.



# Bayesian Methods for Adaptive Models
Bayesian model inference of parameters and model comparison. Hasn't finished because was too difficult.

# Tutorial on Variational Autoencoders
Explained VAE and CVAE. very nice

# Variational inference for Monte Carlo objectives
Using multi sample stochastic lower bound for log likelihood. It can deal with discreate and continuous hidden variables.
