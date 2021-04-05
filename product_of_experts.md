#[Training Products of Experts by Minimizing Contrastive Divergence](https://dl.acm.org/citation.cfm?id=639730)
2002

Extension of work in "Products of Experts". Each expert has to be a little bit more complicated than for example
Gaussian because product of Gaussians is still Gaussian and MoG can recover any complex smooth distribution.
Here author presents that instead of maximizing the likelihood of data which is minimizing the KL divergence between the data distribution
and the model generated distribution we can minimize the difference between the two KL divergences. KL(Q_0||Q_inf) - KL(Q_1||Q_inf) where
Q is p(d|...). The Q_0 is the data distribution, the Q_1 is the distribution after running one full cycle of the Gibbs sampler and 
Q_inf is the distribution of the model at equilibrium. This way the model will be taught to generate data samples and we get rid of
difficult to compute expectation.

#[Products of Experts Welling](http://www.scholarpedia.org/article/Product_of_experts)
2007
PoE act by carving the distribution and MoE by adding probability regions. The nice interpetetion of the maximum likelihood learning
where first term is interpreted as increasing likelihood of the data for the model and decreasing the likelihood where the model
already assigns high probability. Also contains more updated references about PoE.
