# Temporal Convolutional Networks: A Unified Approach to Action Segmentation
Using one model for temporal segmentation of actions in the video data (TCN) instead of complex models composed of CNN and RNN. According to authors this is easier to train and gives better results.
# Deep Learning Book, chapter about CNN
TDNN (Time Delay Neural Networks) are 1D convolutional NN for TS introduced by Land and Hinton in 1988 (The development of the time-delay neural network architecture for speech recognition.)
# Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
https://arxiv.org/abs/1703.07015
Models multivariate time series using combination of CNN and RNN.
Use nice data set of traffic on the motorway with 2 separate pattern daily on weekday and weekly on weekends. (all their data can be found at https://github.com/laiguokun/multivariate-time-series-data)
What is interesting their complicated model does better than benchmark models (like AR, GP, SVR) on data with repetitive patterns (traffic, electricity) but worse on noisy data like exchange rates.
# Convolutional networks for images, speech, and time series
Early description of conv nets. TDNN mentioned as CNN which share weights across one temporal dimension. 
Using probabilistic method ass HMM on the output of the replicated CNN (to handle variable size inputs like sentences) to decode the objects on which given CNN centers on (word, letter). Could we apply our VI HMM on this CNN.
# Going deeper with convolutions
Making CNN more efficient and steel improving results compared to the previous bigger CNNs. This could be good for my commercial domain.
# Network in network
Authors use MLP instead of GLM as the convolution filters. At the last layer instead of the fully connected NN authors use global average pooling which creates feature maps for each category. This mean that for each category we create feature map using convolutions and then each map is averaged before all going into softmax layer.I think this idea can also be applied for time series analysis for pattern detection in sequences.
One by One convolution that can be used for dimensionality reduction.
# Improving neural networks by preventing co-adaptation of feature detectors
Original paper about the dropout. Interesting idea about limiting the L2 norm of the vector of the incoming weights to each unit which allows to start with big learning rate (when updating the weights the norm is too large we simply rescale the weights). Compares dropout to Bayesian NN.
Interestingly the authors mention that CNN filters are good for datasets with roughly stationary statistics because the same filter is applied across all examples in the dataset. 
Something new for me here is response normalization units where activations of the neurons at the same positions from different banks are normalized.
Nice discussion about using linear rectifiers units. We have to initialize weights in a way so the input to the linear rectifier are mostly positive because otherwise learning doesn’t take place. It helps to set constant to some positive value (authors use 1) to get learning of the ground. For weight initialization they use normal with 0 and some variance big enough to get positive inputs (chosen experimentally).
# Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition
HMM - DNN model which combines neural network and HMM. Here NN provides observation probabilities.
# Unsupervised Neural Hidden Markov Models
Performing inference in HMM using NN. Replaces transition and emission matrices with output of the NN. The posteriors are still computed using Baum-Welch algorithm and they use 2 stage EM algorithm. The posteriors are used to weight gradients of the emission and transition which can be computed using NN. It would be interesting to compare it to our model.
Using CNN for observation distribution and RNN (LTSM) for transition model (effectively replacing 1 step Markov with dependence on all previous data) … but does it mean that data have to be fed sequentially and we cannot do stochastic optimization like we do.
Lessons learnt by authors could be important to our previous work because they found that when using NN in unsupervised model it is more important to use dropout and proper weight initialization compared to supervised models.
# A survey of hybrid ANN/HMM models for automatic speech recognition
Such models were developed already in 80s and beginning of 90s. Authors described different combinations if HMM and ANN and different ways of training. Separate (HMM and ANN params are learned separately) and global schemes are described (all parameters are learned together). Are hybrid systems worth considering a research topic or obsolete?
# A Connectionist Approach to Speech Recognition
Integration prior knowledge with learning: constraints on architecture, constraints that guide training, prior knowledge used to initialize free parameters. Different interesting ideas I haven’t heard before like: adding special outputs, smoothing the inputs to help with regularization. Partitioning of large system into smaller networks is described and training it globally and each part separately. Different architectures for processing sequences, more interesing: TDNN, RNN
# Supervised Sequence Labelling with Recurrent Neural Networks
Limitations of RNN: remembering long context (LSTM), only access to the past context (bidirectional RNNs)
Limitations of hybrid systems (combining HMMs with RNNs) mainly comes from limitations of HMMs. 
MD RNNS are better than convolutional NN for warped data (on  MNIST) so it looks it should be better for time series modeling. (Graves notes in this monograph that MDRNN take much more time to train the CNNs)
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
Very important paper if someone wants to extend LSTM because it checks all of its components how performance deteriorates when components are removed. Very nice framework for comparing the model. Not that important: peephole. Can merge input and forget (GRU) without loosing too much accuracy. Gates are important. For most data sets nonlinearities (input/output are important). The most important hyper parameter is learning rate then the size of the hidden layer. Momentum proved be not that important and input noise. Interaction between learning rate and hidden size is the biggest but not much structure so can choose learning rate on smaller network and then use it on bigger networks.
# Recurrent Neural Networks for Missing or Asynchronous Data
Full probabilistic approach where we learn the joint (requires lots of parameters) vs discriminant approach where we learn for example only boundaries (requires fewer parameters). The easiest method when missing data is simply to use unconditional means for missing data. 
Somehow the missing data is filled in by feedback links from hidden and output units into input units.
# Recurrent Neural Networks for Multivariate Time Series with Missing Values
Informative missingness as additional information for the model. Authors here use GRU and apply exponential decaying on input variables and hidden variables. The input are variables, masking bits (missing/available), and time since observation was available. 
Contain references to interesting free medical data with time series records. Interesting experiments that I can redo on my problem.
Computing correlation between missing rate for each variable vs target variable to show that missingness has some information.
# A Solution for Missing Data in Recurrent Neural Networks With an Application to Blood Glucose Prediction
RNN is used to model nonlinear dynamic of the system and KF is used to model linear error model. The models are learned alternatively. I should revisit this paper when learn more about Kalman filter.
# Professor Forcing A New Algorithm for Training Recurrent Networks
Improving long term dependencies in the generated samples. It helps hidden states to stay in the same region as in training during the sampling (they use t-SNE to demonstrate it). Professor forcing  uses GAN approach by have generative model training using teacher forcing and discriminative model classifying if generative is running in the teacher forcing mode.
There are some ad-hoc rules like thresholds used when to use gradients from the discriminant network which doesn’t look too good.
# A learning algorithm for continually running fully recurrent neural  networks
The algorithm presented here allows for the missing data i.e. error is computed only over outputs that are given at the time and 0 otherwise (it shows the the separate problem of missing inputs and missing outputs). They use teacher forcing to improve the results. Each weight depends on the entire recurrent matrix and all errors so it is not local.
# Dynamic Routing Between Capsules
Capsules are routed using Hebbian rules i.e. the connection between capsules on the consecutive layers are enforced if their outputs agree. Capsules are applied in the convolutional way. Don’t fully grasp the architecture so have to revisit it later.
# Transforming Auto-encoders
Network of capsules that learns to transform images. Capsule has recognition and generative units. It has also boolean gate to recognize if entity is on the picture. When applied in CNN way there is no weight sharing?
# Matrix capsules with EM routing
Reach source of references on how to improve CNNs by adding other types of invariances. Only read beginning.
# Neural Turing Machine
Nice experiments tesing if network can learn to copy, use associative memory, model n-gram. Simple differentiable controller which can read and write to memory using location addressing (shift) and associative addressing. Controller learns to output weights over the memory and its output is the output of the model or can be mixed with output of some other net?
# Hybrid computing using a neural network with dynamic external memory
This is the Neural Turing Machine with improved addressing (link Matrix keeping the order in which memory was accessed, removing convolutional shift). The link matrix can be sparse without decrease of the performance. Model is learned using supervised learning and also reinforce algorithms.
# Neural Random-Access Machines
There is a controller, modules (operations), registers. Controller orchestrates operations over registers using operations and the results are written back to the registers.
The circuits are like mini algorithms (subroutine) that are executed step by step. More difficult problems are hard to learn possibly because of weak optimization. Possibly QA can help.
# Adding Gradient Noise Improves Learning for Very Deep Networks
This uses annealed noise added to gradient. Maybe it can be compared to QA. Added gradient noise is similar to simulated annealing. Adding noise only helps with very deep architectures h > 5. For simple FFNN It looks that using good init and grad clipping achieves the same results as the same with grad noise. Grad noise helps when only simple init. But it is still useful when network is complex and we are not sure about proper init. But this is not true for more complex architectures where finding perfect init values is much harder.
# Reinforcement learning neural Turing machines 
Important because continuous-discrete learning. This model is very difficult to train for more complex problems so could be could candidate for QA. 
Curriculum learning is very important to be able to learn. It is also important to decrease the variance of the gradient estimator.
Reinforce is used to construct the cost function but it is simply expectation of the actions over the cumulative rewards.
# Quantum machine learning : what quantum computing means to datamining / Peter Wittek
Started but in the end was too hard to follow. Maybe later.
# Memory networks
Very generic architecture for memory enhanced models. Components I (input),G(generalization i.e. memory access/write), O (output feature from memory given input), R (response). The memory stores all sequences (if words are given there is a model to discover sequences). Then memory is compared to each sequence in order in the test set . Authors use hashing of the memory to imrove performance. The model is applied to QA tasks.
