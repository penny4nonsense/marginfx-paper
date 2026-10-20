# Cover letter — draft

> Draft for submission to *JRSS-B*. Sections marked **[FILL]** need your input.
> The journal requires the cover letter to carry the corresponding author's
> full contact details, a conflict of interest statement, a short running head
> of 50 characters or less, and a disclosure of any AI use.

---

To the Editors, *Journal of the Royal Statistical Society, Series B*

I am submitting **Model-Agnostic Interpretation of Machine Learning via Average
Marginal Effects** for consideration in the Theory and Methods section.

## Significance

[FILL — a paragraph on why the paper matters. The elements are already in your
abstract and introduction, and the strongest version is probably: average
marginal effects are the standard inferential tool in the econometrics of
nonlinear models but have been confined to parametric ones; making the window
the estimand rather than an approximation to a derivative extends them to any
supervised learner, including learners differentiable nowhere; and because the
functional is linear it admits an exact Riesz representation, which yields an
orthogonal score whose bias is exactly a product of two nuisance errors and an
estimator attaining the semiparametric efficiency bound. The practical payoff
is a coefficient table for a black-box model, with standard errors and no
resampling.]

## Corresponding author

- Jason Parker
- [FILL — mailing address]
- jparker588@gmail.com
- [FILL — ORCID, if you have one; optional but useful]

**Running head:** [FILL — 50 characters or fewer. `Model-Agnostic Average
Marginal Effects` is 40.]

## Conflict of interest

[FILL — if none: "The author declares no conflicts of interest." Note that the
`marginfx` package is the author's own software; whether that warrants
mention is your call, but it is the kind of thing an editor would rather see
disclosed than discover.]

## Use of artificial intelligence

In accordance with the journal's policy, I disclose that I used Claude
(Anthropic) extensively in the preparation of this manuscript, the
supplement, and the accompanying code. The assistance fell into four areas.

**Manuscript text.** Claude drafted and revised prose throughout the main
paper and the supplement, and carried out the restructuring of material
between the two documents, including the reduction of the main text to its
present length. The research programme, the estimand, the framework, and all
decisions about what to claim are mine.

**Proofs.** Claude's contribution to the mathematics is concentrated in the
gradient-consistency results of the supplement — the results characterising
what a learner's own gradient information estimates — and not in the
inference theory, which predates this assistance. Specifically, it produced
the current form of the per-tree bound for Mondrian forests, including the
per-leaf charging lemma and the truncated inverse-moment lemma; the
deterministic gradient-modulus lemma underlying the Sobolev consistency
result; and the correction of several defects in earlier versions of those
arguments. It also established, by numerical investigation, that gradient
consistency for greedily grown gradient boosting is not attainable by the
routes available, which is why the boosting theorem is stated for a fixed
symmetric linear learner only. I have checked these arguments and am
responsible for them.

**Code.** Claude wrote the numerical verification scripts in the replication
repository (`verification/`), which support the proofs above; the manuscript
consistency checks (`tools/`); and parts of the table-generation pipeline. It
also assisted with the empirical and simulation code.

**Verification of reported results.** Claude was used to check quoted numbers
in the manuscript against the stored result files, and to check the two
documents for internal consistency and conformity with the journal's
submission requirements.

Claude is not an author of this work and is not listed as one. All
mathematical claims, all reported numbers, and all editorial decisions are
mine, and I am solely responsible for the content of the paper.

[FILL — consider adding here any other AI tools used at any stage, and whether
any of the referee-style critiques that shaped the revisions were themselves
machine-generated. Only you know the full history; the account above covers
what I can attest to.]

## Data and code availability

All three datasets analysed are publicly available. Code reproducing every
numerical result in the paper and supplement is available at
https://github.com/penny4nonsense/marginfx-paper, with download scripts for
the raw data and the environment needed to run them. A permanently archived
version with a DOI will be deposited before publication.

## Suggested referees

[FILL — optional, but the journal notes it is helpful. Natural candidates sit
at the junction of the debiased machine learning and average-derivative
literatures.]

Yours sincerely,

Jason Parker
