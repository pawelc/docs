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
Such models were developed already in 80s and beginning of 90s. Authors described different combinations if HMM and ANN and different ways of training. 
Separate (HMM and ANN params are learned separately) and global schemes are described (all parameters are learned together). 
Are hybrid systems worth considering a research topic or obsolete?
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
# Dynamic Routing Between Capsules
Capsules are routed using Hebbian rules i.e. the connection between capsules on the consecutive layers are enforced if their outputs agree. Capsules are applied in the convolutional way. Don’t fully grasp the architecture so have to revisit it later.
# Transforming Auto-encoders
Network of capsules that learns to transform images. Capsule has recognition and generative units. It has also boolean gate to recognize if entity is on the picture. When applied in CNN way there is no weight sharing?
# Matrix capsules with EM routing
Reach source of references on how to improve CNNs by adding other types of invariances. Only read beginning.
# Neural Random-Access Machines
There is a controller, modules (operations), registers. Controller orchestrates operations over registers using operations and the results are written back to the 
registers.
The circuits are like mini algorithms (subroutine) that are executed step by step. More difficult problems are hard to learn possibly because of weak optimization. 
Possibly QA can help.
# Adding Gradient Noise Improves Learning for Very Deep Networks
This uses annealed noise added to gradient. Maybe it can be compared to QA. Added gradient noise is similar to simulated annealing. 
Adding noise only helps with very deep architectures h > 5. For simple FFNN It looks that using good init and grad clipping achieves the same results as the same with 
grad noise. Grad noise helps when only simple init. But it is still useful when network is complex and we are not sure about proper init. But this is not true for more complex architectures where finding perfect init values is much harder.
# Reinforcement learning neural Turing machines 
This model uses hard attention to address the memory. Most external interfaces are discrete so very useful. It combines backpropagation with reinforce.
Important because continuous-discrete learning. This model is very difficult to train for more complex problems so could be could candidate for QA. 
Curriculum learning is very important to be able to learn. It is also important to decrease the variance of the gradient estimator.
Reinforce ie policy gradient is used to construct the cost function but it is simply expectation of the actions over the cumulative rewards.
They had to hack controller a bit to be able to solve the tasks. They call it direct access controler because it can for example copy directly input to memory
modulated only by variable from the controller.
Can read further how they make method more stable and how to implement it.
# Quantum machine learning : what quantum computing means to datamining / Peter Wittek
Started but in the end was too hard to follow. Maybe later.
# Memory networks
Very generic architecture for memory enhanced models. Components I (input),G(generalization i.e. memory access/write), O (output feature from memory given input), 
R (response). The memory stores all sequences (if words are given there is a model to discover sequences). 
Then memory is compared to each sequence in order in the test set . Authors use hashing of the memory to improve performance. The model is applied to QA tasks.
# Weakly Supervised Memory Networks
The same application as Memory Networs but doesn't hint which intermediate sentences help with answering question hence weak (no supervision of the supporting facts). Like Diffrerentiable Neural Computer they try to keep temporal information for memory access.
They try to fight the local minima by first running the model with all non-linear oprations removed and later when 
there is no improvement they enable non-linearity (they call it linear start).
# Learning to Execute
LSTM that learns what is the output of the short program. The authors also test the LSTM model on copy task and notice that improvement can come from few heuristics like inversing the 
input or doubling it. Authors also notice that learning is only possible when applying their new curriculum learning (mixing simple curriculum  and some random that can select also difficult 
examples like in Reinforcement learning neural Turing machines). It looks that combined curriculum  learning is mandatory and naive one can be worse the data distribution one.
Authors generate programs automatically that meet some criteria  regarding the difficulty. Overall this article seems to be weaker model than for example NTM.
# Recent Advances in Recurrent Neural Networks
Nice review of RNN from the historic models upto the state of the art like memory networks.
# Mastering the game of Go with deep neural networks and tree search
Use value, policy networks together with Monte Carlo Tree Search and other techniques to achieve the best ever AI go player. First policy network is trained supervised on database of games. It is seed to policy network which is learnt using RL with self play. On these self play games value network is trained. 
# Adaptive systems for foreign exchange trading
Technical indicators alone are not enough to make strategy profitable. But if we use market maker's inside info of client flow and order book then we can
create profitable strategy. Authors check cointegration tests for flows and market moves and find the some particular flow like hedge fund and funds can conintegrate
nicely with price action. For sterling the most important flows were institutional and corporate.
# Intraday FX Trading: An Evolutionary Reinforcement Learning Approach in "IDEAL 2002 Intelligent Data Engineering and Automated Learning"
Trading agent that buys/sells/keeps neutral using RL with Q lerarning. The state of the system are signals from the technical indicators. This solution
has poor generalization. Using GA algorithm to select which indicators to use helps much out of sample performance.
# Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks
Value function approach to RL like Q-learning, dynamic programing, TD-learning can have problems in finance because of noise, and nostationary data and the policy 
can change dramarically when value function changes only a bit. Actor/critic is indermediete method when actor is learning policy and critic is learning value function.
The recurrent reinforcment learning  (type of direct reinforcment learning when we learn directly policy functuin) can be better apprach.
Authors setup sharp ratio and system tries to optimise it. Authors try various functions for functionally select decision (what position to have at time t) 
using RNN and LSTM. They use gradient descent to optimize across runs. They find learnt strategies profitable. 
They want to try in the future evolution type optimization and learn multi asset trading agent.
# FX trading via recurrent reinforcement learning
Authors optimize differential sharp ratio. They use simple FF NN with one and two layers. It is found that NN with 1 layer achieves better 
results because possibly works better with noise. It gives hint for the future that when using complex models we need good ways of regularization 
(like dropout or noise in the gradient). Authors also find that strategies are profitable when movement/spread ratios are bigger. 
Authors also train model in the online fashion also during test time.
# Making financial trading by recurrent reinforcement learning
Authors use RRL but not using Sharpe ration but ratio of sum of the positive and negative absolute returns. Authors in the same way compute the ratio 
using moving average. They try to minimize the maximum drawdown.
# Learning to trade via direct reinforcement
Authors argue that direct reinforcement i.e. method learning directly policy is better for problems when agent receives immediate estimates of
incremental performance compared to methods using value function like TD-Learning, Q-Learning. Agent to incorporate trading cost, market impact must have recurrent 
structure. Methods using value function is better when reward is deferred considerably in the future, DR is better when we have signal at each step.
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
# Machine Learning for Trading
Using Q learning (discrete actions and states) for trading function. Author derives what reward function should be used in case of concave utility function. 
Author models the market as random mean reverting process. He also models the market impact. 
# Equity market impact Almgren R, CThum, E Hauptmann and H Li 2005
Modeling temporary and persistent market impact by power functions

# Almgren R and N Chriss 1999 Value under liquidation
Value of the portfolio taking into consideration the cost of the liquidation

# Multiperiod Portfolio Selection and Bayesian Dynamic Models
Multiperiod portfolio optimization using HMM.
# Market Making via Reinforcement Learning
Using TD/Q learning to build market making agent. Authors discover that techniques like tile encoding, eligibility traces and reward function that
discourages holding of positions can improve performance of the model.

# FX Spot Trading and Risk Management from A Market Maker’s Perspective
Nice review of FX markets. Comprehensive way of simulating the environment of FX market maker.
Comparison of performance of market maker without hedging, with hedging and with VaR (using extreme value apprach)

# Uncertainty in Deep Learning
Mapping VI with Stochastic Regularization Techniques like dropout. Should  be implementable for DNC.
There are a lot of references in this work worth reading. It would be nice to read later to the end especially about usage 

# Bayesian Methods for Adaptive Models
Bayesian model inference of parameters and model comparison. Hasn't finished because was too difficult.

# Probabilistic Interpretations of Recurrent Neural Networks
Framing RNN as capable of forward computations of state space models like HMM. Also comparing RNN to particle filters.
Authors perform experiments where RNN predicts hidden state of HMM given data. After training RNN the predicted states were the same as from
hmm forward algo.

# GENERATIVE TEMPORAL MODELS WITH MEMORY

This paper combines stochastic variational inference with memory-augmented recurrent neural networks. 
The authors test 4 variants of their models against the Variational Recurrent Neural Network on 7 artificial tasks requiring long term memory. 
The reported log-likelihood lower bound is not obviously improved by the new models on all tasks but is slightly better on tasks requiring high capacity memory.

# Tutorial on Variational Autoencoders
Explained VAE and CVAE. very nice

# A Recurrent Latent Variable Model for Sequential Data
Added random hidden variables to rnn to model the distribution of data better and in generative way. Using VAE to learn
parameters of the model and aproximate posterior. Got better resylts than standard RNNs.

# A statistical view of deep learning 
DNN = Recursive Generalized Linear Models
Autoencoders as Variational approach to learning generative models.
RNN as state space models with deterministic dynamics.
Maximum likelihood, MAP and overfitting. But they lack transformation invariance.
Thoughts when model is deep.

# A neural network is a monference, not a model
NN as model with inference. Shows that RNN directly recovere inference of HMM (filtering) with the same precision. The same as in 
"Probabilistic Interpretations of Recurrent Neural Networks"

There was a workshop about memomy https://nips2015.sched.com/event/4G4h/reasoning-attention-memory-ram-workshop

# Generating sequences with recurrent neural networks
The RNN generates sequenes by letter. it predict next input given previusly geneated output. Network has skip connections from input to all hidden
layers and from all hidden layers to output.
The network output parametrizes required distribution. Shown nicely how to parametrize network for real  and categorical data.
Not read everything but wortth later.

# Analysis of Recurrent Neural Networks for Probabilistic Modeling of Driver Behavior
Authors experiment with application of RNNs (LSTMS) to build model for behaviours of car drivers in terms of their accelaration
when following other vehicle. This is necessary for contemporary cars so they can build mechanisms to brake autonomously or warn drivers.
The models are later used for simulations. So far people used fixed models (specific formulae with tunable parameters) or NN with history.
It looks that using RNN improves different benchmarks. Authors use fifferent parametrizatations of output densities like mixture of Gaussians according to
"Mixture Density Networks" the same as in "Generating sequences with recurrent neural networks" or piecewise uniform.
It looks that FFNN given enough history coped better or the same as LSTM.

# Variational inference for Monte Carlo objectives
Using multi sample stochastic lower bound for log likelihood. It can deal with discreate and continuous hidden variables.

# Variational Memory Addressing in Generative Models
Generative model enhanced with memory trained using Using multi sample stochastic lower bound for log likelihood. The memory
is addressed by discrete random variable which also have approximate posterior. So it uses hard attention. Tehy mention that their addressing
would be plugin replacement for soft addressing used in "GENERATIVE TEMPORAL MODELS WITH MEMORY" 

# Learning context-free grammars: Capabilities and limitations of a recurrent neural network with an external stack memory
Im this paper from 1992 authors already used RNN with external memory (stack in this case). RNN at each time step receives previous state,
input and value from the stack. The output is the output, state and the categorical action which is used to decide if we should push, pop, or no-op
from the stack. It looks they also used curriculum learning. They called it Neural Network Pushdown Automaton.

# End-To-End Memory Networks
The authors propose differentiable architecture for question answersing. All sentences are encoded into internal representation, the output is encoded
separately. Based on internal representation we create weight vector over the output. Question is also embeded and this embeding is summed with output.
This computation can be chained like in RNN but input at each level is output from the previous layer and all the input sentences. Authors
try different kinds of parameter tieing between layers. The layers represent number of think times before network computes the answer.