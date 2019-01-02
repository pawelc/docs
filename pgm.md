#[Products of Experts](https://ieeexplore.ieee.org/document/819532)
**Product of distributions** can be better than mixture. In mixture each component independently try to explain all dimensions.
In PoE each component (expert) can be responsible for the subset of the dimensions. Each expert can put 0 distribution
on different aspects of the data (constraints) and when product is applied all constraints have to be met. The inference is
easy which comes from the fact that hidden variables of different expert are independent given the data. Where for example
in DAG model explaining away makes the inference difficult. The **generation of fantasy data from PoE is difficult** and normally 
we have to use Gibbs sampling. Restricted Boltzman Machine is type of PoE. The gradient of the log probability of PoE is composed
of the easy to compute term if each expert's likelihood can be easily differentiated and expectation of the the derivative of the loglikelihood 
with respect to the model (coming from the normalization constant) which is difficult but can be approximated.

#[Training Products of Experts by Minimizing Contrastive Divergence](https://dl.acm.org/citation.cfm?id=639730)
Extension of work in "Products of Experts". Each expert has to be a little bit more complicated than for example
Gaussian because product of Gaussians is still Gaussian and MoG can recover any complex smooth distribution.
Here author presents that instead of maximizing the likelihood of data which is minimizing the KL divergence between the data distribution
and the model generated distribution we can minimize the difference between the two KL divergences. KL(Q_0||Q_inf) - KL(Q_1||Q_inf) where
Q is p(d|...). The Q_0 is the data distribution, the Q_1 is the distribution after running one full cycle of the Gibbs sampler and 
Q_inf is the distribution of the model at equilibrium. This way the model will be taught to generate data samples and we get rid of
difficult to compute expectation.

#[Markov properties for acyclic directed mixed graphs]
2003
The paper proves local and global Markov properties for the acyclic directed mixed graph.

# On Contrastive Divergence Learning
Use CD to get biased ML solution and then use slow ML learning to fine tune.
The distribution near the boundary of the simplex are more difficult to model. Research Idea: Show the location of the distribution I model.

#[Binary models for marginal independence]
10.2006
Read only about the section 4 about moebius parametrization.

#[Products of Experts Welling](http://www.scholarpedia.org/article/Product_of_experts)
2007
PoE act by carving the distribution and MoE by adding probability regions. The nice interpetetion of the maximum likelihood learning
where first term is interpreted as increasing likelihood of the data for the model and decreasing the likelihood where the model
already assigns high probability. Also contains more updated references about PoE.

#[The Hidden Life of Latent Variables: Bayesian Learning with Mixed Graph Models](http://www.jmlr.org/papers/volume10/silva09a/silva09a.pdf)
04.2008
Not finished paper about Directed Mixed Graphs.

#[Structured ranking learning using cumulative distribution networks]
Rank complex objects (described by vectors). Ranking means construction preference graph which has directed edge from 
object $i$ to object $j$ if there is preference on $i$ over $j$. 
Each data point consist of preference graph over subset of objects. 
During test time we receive list of objects with respective vectors and we have to return the preference graph
over the given list. Authors propose encoding the problem of ranking object into CDN model which nodes are all pairwise 
ranking between objects. Connections between the nodes in the CDN graph depicts dependencies between rankings.

#[Cumulative distribution networks Graphical models for cumulative distribution functions]
2009
Intro to PhD exam to the Cumulative distribution networks and the derivative-sum-product algorithm..

# [Cumulative Distribution Networks Inference, Estimation and Applications of Graphical Models for Cumulative Distribution Functions](http://www.psi.toronto.edu/publications/2009/PhDdocument_CDNs.pdf)
2009

Huang PhD Thesis. Thorough introduction to CDN. Now should read chapter 4 if have time.

# [Mixed Cumulative Distribution Networks](http://proceedings.mlr.press/v15/silva11a/silva11a.pdf)
31.08.2010

ADMG (Acyclic Directed Mixed Graph) are generalization on Directed graphs to contain bi-directional edges.
The can encode more independence assumptions. It is hard to parametrized likelihood functions to  
encode independence assumptions in ADMG. Authors parametrize cdf using product of CDFs. 
Authors extend Cumulative Density Networks to Acyclic Directed Mixed Graph.
Authors show how to factorize ADMXG into product of subgraphs built on districts i.e. connected sets of variables by only 
bidirected connections and the parents of such a district.

# [Exploiting Copula Parameterizations in Graphical Model Construction and Learning]
2011
Ricardo's take on ADMG, CDN with copulas. Nice thorough presentation.

# [Cumulative distribution networks and the derivative-sum-product algorithm: Models and inference for cumulative distribution functions on graphs](http://www.jmlr.org/papers/volume12/huang11a/huang11a.pdf)
Cumulative distribution networks and the derivative-sum-product algorithm
2011

Another class of PGM is CDN (Cumulative Distribution Network) introduced in \citep{HuangJim2012Cdna}. 
They enable to model joint distribution functions in terms of factors of cumulative distribution functions on subset 
of variables. This removes the need for latent variables. It also removes the requirement for the summation/integration 
during inference. In this model we can use limit/differentiation of the CDFs \citep{HuangJc2011CDNa} to perform the same 
operations. Additionally we can ask inference queries that often are difficult to compute with other probabilistic graphical 
models like $F(\bm{x}_A|\bm{x}_B)$. 
Using these operations we can compute $F(\mathbf{x}_A | \omega(\mathbf{x}_B))=\frac{F(\mathbf{x}_A,\mathbf{x}_B)}{F(\mathbf{x}_B)}$
, $F(\mathbf{x}_A | \mathbf{x}_B)$, $P(\mathbf{x}_A | \omega(\mathbf{x}_B))$ and $P(\mathbf{x}_A | \mathbf{x}_B)$ 
where $\omega(\mathbf{x}_B)$ denotes event \{$\mathbf{B} \leq \mathbf{b}$\}. The CDN network is depicted as undirected 
Factor Graph which has bipartite structure with variable nodes and function nodes. It is proven in \citep{HuangJc2011CDNa}
 that if all factors are CDFs then the entire model encoded as product of factors is also a CDF. Authors also prove different 
 characteristics of the CDNs like marginal and conditional independence. The CDNs when marginalized over separation variables 
 create two independent subset of variables. 
 The CDN model has its own version of message passing called derivative-sum-product that allows efficient computation 
 of the CDF mixed derivatives \citep{HuangJc2011CDNa}. This operations result in computing probability function. 
 The algorithm proceeds similarly to the regular sum-product algorithm for factor graphs but instead summation we use 
 differentiation of the local factors. It can be proven \citep{HuangJc2011CDNa} that when algorithm completes at the root 
 of the tree and we differentiate resulting message with respect to the root's variable we end up with joint probability 
 for the model.
 
 #[Bayesian inference in cumulative distribution fields]
 Author uses auxliary latent models in CDF for bayessian learning. This allows to construct complicated copula models
 out of factors (simpler copulas). He uses augmentation of the marginals to construct the copula out of product of factors
 which are copulas themselves.
 There are different methods of using hidden models to help with doing inference in CDN models.
 
 #[Ricardo Silva - Bayesian Inference in Cumulative Distribution Fields](https://www.youtube.com/watch?v=GkEZw3xTQZw)
 Ricardo on CDNs and extending them with latent variables to perform message passing in the extended diagram. Comparing 
 CDNs (product of CDFs which is itself a CDF) to CDF (Cumulative Distribution Fields), which special way of creating 
 factors of copulas is also copula. (when simply multiplying copula functions we generally do not end up with copula because
 marginals are not uniform anymore).
 
#[Deep Exponential Families]
2014
Using different distributions from exponential families  for latent variables we can recover different models.
 Bernoulli latent variables recover the classical sigmoid belief network
Gamma  latent variables give something akin to deep version of nonnegative matrix factorization
Gaussian latent variables lead to the types of models that have recently been explored in the context Deep Exponential Families of computer vision
Using sparse gamma (shape less than 1) to model variables and weights
The explaining away makes inference harder in DEFs then in RBMs but forces a more parsimonious representation where similar features compete to explain the data rather than work in tandem (possibly read  [12, 1] references)
Somewhat related, we find sigmoid DEFs (with normal weights) to be more difficult to train and deeper version perform poorly (worth to check why)
 
