#[Markov properties for acyclic directed mixed graphs]
2003
The paper proves local and global Markov properties for the acyclic directed mixed graph.

#[Binary models for marginal independence]
10.2006
Read only about the section 4 about moebius parametrization.

#[The Hidden Life of Latent Variables: Bayesian Learning with Mixed Graph Models](http://www.jmlr.org/papers/volume10/silva09a/silva09a.pdf)
04.2008
Not finished paper about Directed Mixed Graphs.

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
 
