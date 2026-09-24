# Model design

## Team strengths

Expected home npxG = home baseline × home attack × away defensive vulnerability. Expected away npxG uses the away baseline and the opposing team factors. A larger defense factor means **more conceded production**. Team factors have geometric mean one for an identifiable scale.

Fit the means using Poisson quasi-likelihood for continuous npxG, weighted by `2 ** (-age_days / half_life)`, with L2 penalization on log ratings. The objective does not assume npxG is an integer count. `shrink_k` controls the penalty; it is not a count of imaginary matches. Optimization must converge and remain within numerical limits. The code includes legacy and normalized penalty scaling; replay uses legacy scaling for continuity with the original reference.

## From npxG to outcomes

A separate home/away ratio of weighted historical goals to weighted historical npxG converts production to goal intensity. The simple replay uses the same half-life for ratings and conversion. Research variants with independent conversion windows and mixed historical goals/npxG targets are not implemented by this release's CLI.

Poisson goal distributions form a score matrix. Dixon–Coles rho adjusts 0–0, 0–1, 1–0 and 1–1 probabilities; rho=0 gives independent Poisson. Invalid negative corrections are rejected rather than clipped. The score support expands to limit omitted tail mass. Win/draw/loss and totals are sums over the matrix.

## Evaluation

For every prediction day, fit only matches before midnight UTC. Estimate conversion and historical-frequency benchmark from that same past. A minimum-history warmup is skipped. Never use the target's observed npxG as a predictor. The included replay is retrospective: it cannot establish whether the provider's data was available unchanged at the original cutoff.

RPS evaluates cumulative category probability errors; lower is better. Calibration tables compare average probability with observed frequency by outcome and bin. Small bins are descriptive and cannot establish reliable calibration. Passing numerical tests is evidence of implementation correctness, not forecasting superiority.

## Research extension: XGBoost

The archived challenger used baseline goal intensities, exposure, lagged npxG for/against, deep completions, a PPDA-based pressing feature, rest, and missingness indicators. Its six-setting grid varied depth (1/2/3) and trees (80/180); preceding-season validation selected by RPS. The included protocol and summary retain the original experiment. No challenger training code or historical feature matrix is claimed to be runnable in this minimal core release.
