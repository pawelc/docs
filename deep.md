# Scaling Up Deep Learning, Bengio talk, August 2014
Log likelihood for NN: we can estimate P(Y|X) by parametrizing P(Y|X)=P(Y|omega=f_theta(X))
Loss = -logP(Y|X). It can be Gaussian or multi nuli.
To build bigger models:
use sparsity
partial computation in NN
Issues with backprop: 
composing sigmoids crates function which is highly non linear i.e. flat at most places and with big changes at few places (big derivative)
Using backprop to learn generative models: variational autoencoder.
