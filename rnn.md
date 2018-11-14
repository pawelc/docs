# Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
https://arxiv.org/abs/1703.07015
Models multivariate time series using combination of CNN and RNN.
Use nice data set of traffic on the motorway with 2 separate pattern daily on weekday and weekly on weekends. (all their data can be found at https://github.com/laiguokun/multivariate-time-series-data)
What is interesting their complicated model does better than benchmark models (like AR, GP, SVR) on data with repetitive patterns (traffic, electricity) but worse on noisy data like exchange rates.
# Supervised Sequence Labelling with Recurrent Neural Networks
Limitations of RNN: remembering long context (LSTM), only access to the past context (bidirectional RNNs)
Limitations of hybrid systems (combining HMMs with RNNs) mainly comes from limitations of HMMs. 
MD RNNS are better than convolutional NN for warped data (on  MNIST) so it looks it should be better for time series modeling. 
(Graves notes in this monograph that MDRNN take much more time to train the CNNs)
# A Critical Review of Recurrent Neural Networks for Sequence Learning
RNN has similar representational capacity as FHMM because the number of states grows exponentially with number of neurons in the hidden layer.
# Long short-term memory
LSTM deals well with distributed input, noise, real input, considerably separated input. Can be applied to problems where integration is required or we need non-integrative behaviour (multiplication example). Paper shows many artificial examples how to test behavior of the algorithm. It shows that LSTM can solve problems that weren’t solved by previous architectures. The paper uses truncated backprop and not BPTT.
# Recurrent nets that time and count
Peephole connections are the key to learn timing between events. Longer delays require longger training.
# Learning to Forget: Continual Prediction with LSTM
Introduces forget gate to be able to process continuous input stream. Otherwise the internal state could blow up and cannot be properly reused through time. Interesting examples for continuous prediction problem.
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
Nice short python code implementing learning and testing of RNN on char level language model and review.
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
some simple overview
# https://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html
a lot of new concepts like vanishing sensitivity = vanishing gradient. Regular LSTMs don’t work. We have to improve it by:
normalizing the state (divide state by sqrt(var + 1))
GRU (forget and input gate are merged into update gate)
pseudo LSTM (state and output are not the same, output goes through squashing function)
Should reread it.
# LSTM: A Search Space Odyssey
Very important paper if someone wants to extend LSTM because it checks all of its components how performance deteriorates when components are removed. 
Very nice framework for comparing the model. Not that important: peephole. Can merge input and forget (GRU) without loosing too much accuracy. 
Gates are important. For most data sets nonlinearities (input/output are important). 
The most important hyper parameter is learning rate then the size of the hidden layer. Momentum proved be not that important and input noise. 
Interaction between learning rate and hidden size is the biggest but not much structure so can choose learning rate on smaller network and then use it on bigger networks.
# Recurrent Neural Networks for Missing or Asynchronous Data
Full probabilistic approach where we learn the joint (requires lots of parameters) vs discriminant approach where we learn for example only boundaries 
(requires fewer parameters). The easiest method when missing data is simply to use unconditional means for missing data. 
Somehow the missing data is filled in by feedback links from hidden and output units into input units.
# Recurrent Neural Networks for Multivariate Time Series with Missing Values
Informative missingness as additional information for the model. Authors here use GRU and apply exponential decaying on input variables and hidden variables. 
The input are variables, masking bits (missing/available), and time since observation was available. 
Contain references to interesting free medical data with time series records. Interesting experiments that I can redo on my problem.
Computing correlation between missing rate for each variable vs target variable to show that missingness has some information.
# A Solution for Missing Data in Recurrent Neural Networks With an Application to Blood Glucose Prediction
RNN is used to model nonlinear dynamic of the system and KF is used to model linear error model. The models are learned alternatively. I should revisit this paper when learn more about Kalman filter.
# Professor Forcing A New Algorithm for Training Recurrent Networks
Improving long term dependencies in the generated samples. It helps hidden states to stay in the same region as in training during the sampling 
(they use t-SNE to demonstrate it). Professor forcing  uses GAN approach by have generative model training using teacher forcing and discriminative model classifying if generative is running in the teacher forcing mode.
There are some ad-hoc rules like thresholds used when to use gradients from the discriminant network which doesn’t look too good.
# A learning algorithm for continually running fully recurrent neural  networks
The algorithm presented here allows for the missing data i.e. error is computed only over outputs that are given at the time and 0 otherwise (it shows the the separate problem of missing inputs and missing outputs). They use teacher forcing to improve the results. Each weight depends on the entire recurrent matrix and all errors so it is not local.
# Probabilistic Interpretations of Recurrent Neural Networks
Framing RNN as capable of forward computations of state space models like HMM. Also comparing RNN to particle filters.
Authors perform experiments where RNN predicts hidden state of HMM given data. After training RNN the predicted states were the same as from
hmm forward algo.
# A Recurrent Latent Variable Model for Sequential Data
Added random hidden variables to rnn to model the distribution of data better and in generative way. Using VAE to learn
parameters of the model and aproximate posterior. Got better resylts than standard RNNs.
# Generating sequences with recurrent neural networks
The RNN generates sequenes by letter. it predict next input given previusly geneated output. Network has skip connections from input to all hidden
layers and from all hidden layers to output.
The network output parametrizes required distribution. Shown nicely how to parametrize network for real  and categorical data.
Not read everything but worth later.

# Analysis of Recurrent Neural Networks for Probabilistic Modeling of Driver Behavior
Authors experiment with application of RNNs (LSTMS) to build model for behaviours of car drivers in terms of their accelaration
when following other vehicle. This is necessary for contemporary cars so they can build mechanisms to brake autonomously or warn drivers.
The models are later used for simulations. So far people used fixed models (specific formulae with tunable parameters) or NN with history.
It looks that using RNN improves different benchmarks. Authors use fifferent parametrizatations of output densities like mixture of Gaussians according to
"Mixture Density Networks" the same as in "Generating sequences with recurrent neural networks" or piecewise uniform.
It looks that FFNN given enough history coped better or the same as LSTM.
# Bidirectional Recurrent Neural Networks as Generative Models - Reconstructing Gaps in Time Series
Authors use BRNN to model sequence of binary observations. They model smoothing distribution.
The model purpose is to fill in the gaps in multidimensional time series data. Gaps are of consecutive vectors which depend on each other so it is difficult
problem. Authors use 2 methods Generative Stochastic Network and Neural Autoregressive Density Estimator. In GSN they fill in the gaps with random value and then
by sampling the missing values at random from learned distribution of P(x_t | {x_o o!= t}. 
They compare their methods to few baseline models like Bayesian MCMC(method to compute smoothing distribution using forward RNN), 
simply computing unnormalized probability for each value and then normalizing using RNN.
The first experiment is using categorical data taken as all Wikipedia text. What is interesting the use only error from the last elements of the sequence in case
of RNN (for example 3/4) or middle elements in case of BRNN. They do it to only backpropagate error with long time dependencies.
To train with NADA they add one dimension as missing values indicator. Mark 5 consecutive step every 25 steps as gaps. Error is only propagated from the gaps.
They uniformly sample within the gaps which one of them to set actually to missing.
# On the difficulty of training recurrent neural networks
The crucial property for learning time series is ability to “discover temporal correlations between events that are possibly far away from each other in the data”
