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