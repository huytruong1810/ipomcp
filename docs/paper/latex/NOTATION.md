# Notation and voice (not part of the paper)

The paper follows Gmytrasiewicz & Doshi (2005), "A Framework for Sequential Planning in Multi-Agent Settings", JAIR 24:49–79.

**Notation.**
- I-POMDP_i = ⟨IS_i, A, T_i, Ω_i, O_i, R_i⟩.
- IS_i = S × M_j and A = A_i × A_j.
- Types: θ_j = ⟨b_j, θ̂_j⟩, with the frame θ̂_j = ⟨A_j, Ω_j, T_j, O_j, R_j, OC_j⟩.
- Subintentional models SM_j; level 0 is the no-information model (uniform over A_j).
- Finite nesting: IS_{i,l} = S × M_{j,l−1}, Θ_{j,l} = {⟨b_{j,l}, θ̂_j⟩}, and M_{j,l} = Θ_{j,l} ∪ M_{j,l−1}.
- Belief update: b_i^t = SE_{θ_i}(b_i^{t−1}, a_i^{t−1}, o_i^t), with β the normalizer and τ_{θ_j} the indicator of j's own update.
- Pr(a_j | θ_j) = 1/|OPT(θ_j)| on OPT(θ_j).
- Values: U^H(θ_i) = max_a Q^H(θ_i, a), with ER_i(is, a_i) = Σ_{a_j} R_i(is, a_i, a_j) Pr(a_j | m_j).
- Ours, added on top: SẼ^N (compressed update), Q⋆ (scored values), [Q⁻, Q⁺] (certified intervals), b_h (belief at private history h), and W_r and ℓ_r, u_r (return ranges and bounds).

**Voice.**
- First-person plural; "agent i" and "agent j".
- Intuition before the math.
- Definitions and propositions set off formally, with proofs or proof sketches.
- The multi-agent tiger game as the running example.
- No development history, phase names or internal result tags.
