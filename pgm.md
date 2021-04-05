#[Fusion, propagation, and structuring in belief networks]()
1986

Connection between human knowledge amd graphical models. Introduction of noisy-or. Pearl introducing belief networks.
Did not finish reading it but maybe it is worth. Noisy Or is example of Independence of casual influence

#[Probabilistic Diagnosis Using a Reformulation of the Internist-1/ QMR Knowledge Base II. Evaluation of Diagnostic Performance]()
1991

Authors develop decision theoretic version of the Quick Medical reference which is decision-support tool for 
diagnosis in internal medicine. The assumptions are marginal independence of diseases, conditional independence of 
findings given any hypothesis of diseases, causal independence of the influence of multiple diseases on a single finding, 
and binary-valued findings and diseases. Not read very thoroughly. 

#[A Generalization of the Noisy-Or Model]()
1993

An extension of the noisy-or to multiple values and any function other than or.
Nice derivation of the P(output/input) probability for generic deterministic function and discrete output/input where
each input can be inhibited to arbitrary state with some probability (not only false like in noisy-or case).
Only inference, there is nothing about parameter learning.
Here the implementation was translating the noisy or model into table representation and from here used regular
belief propagation algorithm.

#[Parameter adjustment in Bayes networks. The generalized noisy OR–gate]()
1993

Sequential updating of parameters of the bayessian network (generic and example given for noisy or). Sequential meaning
that parameters are updated given next example (can be partial).

#[Bayesian Network Classifiers]()
1997

Extending Naive Bayes classifier into TAN (Tree Augmented Naive Bayes). They look for the structure of the optimal
Bayesian Network using MDL score function (the edges connecting A_i s and A_s with C are 
searched over using local modifications). They show that MDL can be not optimal for learning classifier using
BN because the MDL is dominated by marginal probability of the attributes and the classifier really use P(C|attributes)
(which remains constant no matter how many attributes are there). They show it by empirical study that learning
unrestricted BN using MDL score gives better (then Naive Bayes) results when there are few attributes but much worse when
number of attributes grows. The authors find that the problem is the score function which causes the classifier learnt can
create connectivity such C given its Markov blanket is independent of some of the attributes that are still crucial for good
classification.
Authors notice that to fix this issue we could optimize Conditional Loglikelihood P(C|Attributes) but then the parameters'
estimator is different then when using LL.
Authors propose extension of the Naive Bayes - Augmented NB by keeping the basic structure of all edges between C and all 
A_i's but also allowing for connections between attributes, this way the class depends on all attributes but also we add 
dependency between attributes when C is given. This models the case when one attribute influence on the class depends on
other attribute and not like in NB all attribute influence class in isolation. The TAN model is NB with additional edges 
between attributes where one attribute can have upto one augmented edge pointing at it. This constrain is applied so the
model can be learnt efficiently. To construct the best tree containing attributes the authors use existing procedure
to span tree over complete graph using maximum weight spanning tree algorithm but using conditional mutual information i.e.
I(A_1;A_2|C). 
Authors also propose method of constructing the TAN model per each class - the multinet Bayes Classifier.

#[Approximating the Noisy-Or Model by Naive Bayes]()
1998

Stopped reading it because was confessed by the notation.

#[Multiplicative Factorization of Noisy-Max]()
1999

Noisy OR and its generalization - Noisy Max were created for efficient knowledge acquisition. This paper shows algorithm
for efficient inference in graphs containing such nodes. 
They present existing ways of encoding max decreasing the complexity of max operator like parent divorcing or temporal
transformation. Authors show previous factorizations like additive ones and introduce their own: multiplicative
factorization.

#[An efficient factorization for the noisy MAX]()
2003

New parametrization using CDFs of the noisy max which is generalization of noisy or (both are for graded/ordinal variables ).
We aim that inference in this canonical model is linear in the number of the parents (and not exponential as in the 
regular CPT implementation). The factorization of the probability contains CDFs for each cause separately and special indicator
variable which says which cdf to use to compute pdf. The show modification of the variable elimination and junction
tree to accommodate this dummy variable and new potential. The junction tree is created slightly differently than 
the regular BN by not moralizing parents of the the noisy max and by introduction of the one dummy/auxiliary variable. We can 
run regular message passing algorithm on resulting graph.

#[Products of Experts](https://ieeexplore.ieee.org/document/819532)
**Product of distributions** can be better than mixture. In mixture each component independently try to explain all dimensions.
In PoE each component (expert) can be responsible for the subset of the dimensions. Each expert can put 0 distribution
on different aspects of the data (constraints) and when product is applied all constraints have to be met. The inference is
easy which comes from the fact that hidden variables of different expert are independent given the data. Where for example
in DAG model explaining away makes the inference difficult. The **generation of fantasy data from PoE is difficult** and normally 
we have to use Gibbs sampling. Restricted Boltzman Machine is type of PoE. The gradient of the log probability of PoE is composed
of the easy to compute term if each expert's likelihood can be easily differentiated and expectation of the the derivative of the loglikelihood 
with respect to the model (coming from the normalization constant) which is difficult but can be approximated.

#[Markov properties for acyclic directed mixed graphs]
2003
The paper proves local and global Markov properties for the acyclic directed mixed graph.

#[A new algorithm for maximum likelihood estimation in Gaussian models for marginal independence]()
Paper about bi-directed graphs with Gaussian dependencies. Here each vertex is colider so when any path between 2 nodes
contains all the vertices in conditioning set than the nodes are m-connected otherwise they are m-separated. The global markov property
is fulfilled by distribution for G if X_A independent of X_B given X_S when A is m-separated from B given S.
We can create bidirected graph from the DAG with V(isible) and H(idden) nodes if V has no children by connecting V nodes which have common
ancestors then such created graph composed only from V nodes has V_1 m-separated from V_2 given S <==> V_1 d-separated from V_2 given S in the DAG 
graph. Couldn't understand the algo :(.  

# On Contrastive Divergence Learning
Use CD to get biased ML solution and then use slow ML learning to fine tune.
The distribution near the boundary of the simplex are more difficult to model. Research Idea: Show the location of the distribution I model.

#[Binary models for marginal independence]())
10.2006
Read only about the section 4 about moebius parametrization.

#[Noisy‐or classifier]()
2006

The noisy OR classification is equivalent to the logistic regression (authors provide proof for this). 
But we can train noisy or using EM and therefore have
missing data. Authors derive efficient computation of marginal given the evidence using auxiliary variable method to 
translate noisy-OR into regular BN (like in "An efficient factorization for the noisy MAX"), the auxiliary variable
selects CDFs to compute probability. This is used in EM algorithm to estimate maximum likelihood parameters 
from incomplete data. One of the problems is not identifiability which can be addressed by constraining the model. For
example authors propose monotonicity constraint (the instance not belonging to class is more probable when attribute is missing). 
Authors compare noisy or to other classifiers like logistic regression, decision tree, naive bayes, SVM on subset of 
euters-21578 data.

#[Relational learning with Gaussian processes]()
2006

#[Hidden Common Cause Relations in Relational Learning](https://papers.nips.cc/paper/3276-hidden-common-cause-relations-in-relational-learning.pdf)
2007

Using relation between objects to improve classification accuracy. 


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
Intro to PhD exam to the Cumulative distribution networks and the derivative-sum-product algorithm.

# [Cumulative Distribution Networks Inference, Estimation and Applications of Graphical Models for Cumulative Distribution Functions](http://www.psi.toronto.edu/publications/2009/PhDdocument_CDNs.pdf)
2009

Huang PhD Thesis. Thorough introduction to CDN. Now should read chapter 4 if have time.

#[A factorization criterion for acyclic directed mixed graphs]()
2009
The ADMG contain the bidirected and directed edges. Authors show how to efficiently create factors for for a graph so the distribution
created meets the global markov property for the given graph. A lot of definitions about the structural properties of ADMG.
NRF

#[Exact inference and learning for cumulative distribution functions on loopy graphs]()
Authors show how to efficiently **compute mixed derivatives** in the **CDN** network which is not necessarily a tree.
This is necessary to compute the likelihhod and to estimate the parameters. They show recursive (dynamic programming)
formula to organize computations. This procedure is efficient for CDNs which are composed of many factors where each factor depend on small 
subset of variables. Then we need to split at each time the graph to two partitions and combine derivatives for each partition 
in a recursive fashion.

# [Mixed Cumulative Distribution Networks](http://proceedings.mlr.press/v15/silva11a/silva11a.pdf)
31.08.2010

ADMG (Acyclic Directed Mixed Graph) are generalization on Directed graphs to contain bi-directional edges.
The can encode more independence assumptions. We don't have to specify hidden variables (common causes), they are 
implicitly modelled as bi-directed edges.  
It is hard to parametrize likelihood functions to encode independence assumptions in ADMG. 
Authors parametrize cdf using product of CDFs. They extend Cumulative Density Networks to Acyclic Directed Mixed Graph.
Authors show how to factorize ADMG into product of subgraphs built on districts i.e. connected sets of variables by only 
bi-directed connections and the parents of such a district.

#[Maximum-likelihood learning of cumulative distribution functions on graphs]()
2010
In this paper authors propose Gradient Derivative Product Algorithm to estimate parameters of the CDN model. This is next algorithm 
after Derivative Sum Product algo for inference. The CDNs have equivalence in terms of the marginal independence to
bidirected models or directed models where bidirected models' links are replaced with hidden variables and links from them to variables previously
linked by bidirected edge. Haven't understood the algo because it requires understanding of DSP. Then authors run experiments on
2 data sets (rainfalls and epidmology) and compare to other models like MRF and bidirected model.

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
 
 #[Deep Exponential Families]
2014
Using different distributions from exponential families  for latent variables we can recover different models.
 Bernoulli latent variables recover the classical sigmoid belief network
Gamma  latent variables give something akin to deep version of nonnegative matrix factorization
Gaussian latent variables lead to the types of models that have recently been explored in the context Deep Exponential Families of computer vision
Using sparse gamma (shape less than 1) to model variables and weights
The explaining away makes inference harder in DEFs then in RBMs but forces a more parsimonious representation where similar features compete to explain the data rather than work in tandem (possibly read  [12, 1] references)
Somewhat related, we find sigmoid DEFs (with normal weights) to be more difficult to train and deeper version perform poorly (worth to check why)
 
 #[Bayesian inference in cumulative distribution fields]
 2015
 Author uses auxliary latent models in CDF for bayessian learning. This allows to construct complicated copula models
 out of factors (simpler copulas). He uses augmentation of the marginals to construct the copula out of product of factors
 which are copulas themselves.
 There are different methods of using hidden models to help with doing inference in CDN models.
 
 #[Ricardo Silva - Bayesian Inference in Cumulative Distribution Fields](https://www.youtube.com/watch?v=GkEZw3xTQZw)
 2015
 Ricardo on CDNs and extending them with latent variables to perform message passing in the extended diagram. Comparing 
 CDNs (product of CDFs which is itself a CDF) to CDF (Cumulative Distribution Fields), which special way of creating 
 factors of copulas is also copula. (when simply multiplying copula functions we generally do not end up with copula because
 marginals are not uniform anymore). 

#[Joint Distributions for TensorFlow Probability]()
2020

Declaring bayesian models in TFP.  