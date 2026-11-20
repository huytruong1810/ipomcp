# Related work

What earlier methods do, as their own texts state it, and how this code differs. Every entry was read in the source named beside it; a claim that rests on a secondary source says so.

## Online and approximate I-POMDP planners

| Method | Source | Belief and other agent | Search | Checked against an exact solution? |
|---|---|---|---|---|
| Interactive particle filter with reachability-tree sampling (IPF + RTS) | Doshi & Gmytrasiewicz, JAIR 34, 2009 ([arXiv 1401.3455](https://arxiv.org/abs/1401.3455)) | N sampled particles per level, weighted by the observation likelihood | expands K sampled observations per node | by average reward: approximate and exact policies at levels 1 and 2, horizons 2 and 3 (Sec. 10.1, Fig. 15); a probabilistic error bound at level 1 (Prop. 3) |
| POMCP adapted to I-POMDPs | Hula, Montague & Dayan, PLoS Comput. Biol. 11(6), 2015 ([doi](https://doi.org/10.1371/journal.pcbi.1004254)) | partner models inside the state; nested inference while building each tree | POMCP | no; used to fit human play in a trust game. The paper does not name the method |
| I-POMCP_O | Eck, Shah, Doshi & Soh, AAAI 2020 ([arXiv 1911.08642](https://arxiv.org/abs/1911.08642)) | many agents of known types in open systems, I-POMDP-Lite and I-POMDP models | MCTS (the name I-POMCP is theirs; POSGGym-Baselines ships a simplified "I-POMCP" derived from it) | no; Theorem 2 bounds regret linearly in the error of the configuration probabilities |
| I-POMDP Lite | Hoang & Low, IJCAI 2013 ([arXiv 1304.5159](https://arxiv.org/abs/1304.5159)) | the other agent is a nested MDP: it is assumed to observe the state | value iteration | no; Theorem 4 bounds the policy loss linearly in the prediction error |
| I-NTMCP | Schwartz, Zhou & Kurniawati, IROS 2022 ([authors' copy](https://rdl.cecs.anu.edu.au/papers/iros22_ntmcp.pdf)) | unweighted particles of history-states (state and joint history); one fixed level per agent; needs only a generative model | one belief tree per nesting level, built bottom-up | no; convergence "given the true belief" and a fixed opponent policy, derived from POMCP's proofs (Sec. III) |
| POTMMCP | Schwartz, Kurniawati & Hutter, 2023 ([arXiv 2306.06067](https://arxiv.org/abs/2306.06067)) | a finite set of fixed policies (types) of the other agent | MCTS with a meta-policy | not an I-POMDP solver |
| IPOMDP-Net | Han & Gmytrasiewicz, AAAI 2019 ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4562)) | an I-POMDP model with learned (randomly initialized) functions; level 0 a pretrained QMDP-net | a QMDP planner in a differentiable network | no; QMDP by design |

## Offline and exact I-POMDP solutions

- Exact solutions group the other agent's behaviorally equivalent models into classes (Rathnasabapathy, Doshi & Gmytrasiewicz, AAMAS 2006; the publisher's copy was not accessible, and the attribution is taken from Zeng & Doshi 2012, p. 212). Behavioral equivalence: two models of j are equivalent when their solutions coincide (Zeng & Doshi, JAIR 43, 2012, Def. 1, [arXiv 1401.4600](https://arxiv.org/abs/1401.4600)).
- Interactive point-based value iteration (Doshi & Perez, AAAI 2008) bounds the error of pruning belief points, and states that it is "unable to usefully bound the error due to approximately solving other's models" (p. 67). Its evaluation compares with the particle filter, not with exact solutions.
- Interactive bounded policy iteration (Sonu & Doshi, AAMAS 2012; journal version JAAMAS 2015) compares with I-PBVI by controller size, time and reward, at levels 1 to 4.
- I-DIDs with model equivalences (Zeng & Doshi 2012) give the UAV domain's sensor and reward values used here (Appendix B.3; `docs/DOMAINS.md`).

## Bound-directed search

- **HSVI** (Smith & Simmons, UAI 2004, [arXiv 1207.4166](https://arxiv.org/abs/1207.4166)) selects the action with the largest upper bound (IE-MAX, after Kaelbling 1993) and the observation with the largest probability-weighted excess uncertainty; an interrupted run acts on its lower bound, with regret at most the root width.
- **Online heuristic search** (Ross, Pineau, Paquet & Chaib-draa, JAIR 32, 2008, [arXiv 1401.3436](https://arxiv.org/abs/1401.3436)): AEMS and its finite-time guarantee (Theorem 3.2), branch-and-bound pruning (Sec. 3.2), the stopping rules (Sec. 3.4), and HSVI's descent used online, stopping at a fringe node (Sec. 3.4.4). The gap-directed rule `outcome="gap"` with `root_rule="open"` is this online form of HSVI's descent.
- **BRTDP** (McMahan, Likhachev & Gordon, ICML 2005) draws the next outcome in proportion to its probability-weighted gap (the planner's `outcome="gap"` takes the largest one, as HSVI does), and returns the greedy policy of the pessimistic bound.
- **Fast informed bound** (Hauskrecht, JAIR 13, 2000, [arXiv 1106.0234](https://arxiv.org/abs/1106.0234)) and the tighter informed bound (Krale, Koops, Junges, Simão & Jansen, AAMAS 2025, [arXiv 2502.06523](https://arxiv.org/abs/2502.06523)) are defined for one infinite-horizon POMDP; `planning.frontier_bounds` adapts the fast informed bound to steps to go, with the other agent's later actions maximized (THEORY.md, "Frontier bounds").
- **Deterministic bounds for online POMDP planning** (Barenboim & Indelman, NeurIPS 2023, [arXiv 2310.01791](https://arxiv.org/abs/2310.01791)).
- **Two-sided bounds in partially observable games**: HSVI for zero-sum one-sided POSGs (Horák, Bošanský, Kovařík & Kiekintveld, [arXiv 2010.11243](https://arxiv.org/abs/2010.11243)) solves for equilibria against an opponent who sees the state; the bounds here are for a best response to nested models of an opponent who does not.
- **Optimistic planning** (Hren & Munos, EWRL 2008; Buşoniu & Munos, AISTATS 2012) also explores by upper bounds and returns the action with the largest lower bound.

## Depth of reasoning

- **Cognitive hierarchy** (Camerer, Ho & Chong, QJE 119(3), 2004): step 0 randomizes; step k best-responds to a normalized Poisson distribution over steps 0 to k − 1; about 1.5 steps fit many games. The uniform prior over lower levels used here is a different choice; both satisfy the nesting condition of "Dominance at the boundary level" (THEORY.md).
- **Agent-based theory of mind** (de Weerd, Verbrugge & Verheij, AIJ 199-200, 2013): first- and second-order agents clearly outperform more limited opponents; a third order adds little.
- **Human mentalizing** (Devaine, Hollard & Daunizeau, PLoS Comput. Biol. 10(12), 2014): sophisticated recursive models explain human choices in repeated games only when the game is framed as social.

## How this code differs

- Nested beliefs are exact or compressed-exact updates, so expected rewards and observation laws are exact at every node, and the tree is rescored with the exact observation law.
- Every certified decision carries intervals that contain the exact values of the searched model (with exact beliefs and resolved modeled levels, of the exact I-POMDP), and decisions are checked one by one against the exact I-POMDP solver.
- The agent holds uncertainty over the other agent's level (priors with level-0 mass), where I-NTMCP and I-POMDP Lite fix one level.
- The cost: it needs explicit transition, observation and reward functions, while I-NTMCP needs only a generative model.
