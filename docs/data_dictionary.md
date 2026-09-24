# Match data contract

One row is one completed match. One evaluation input must represent a single competition. Use UTF-8 CSV. All predictions use earlier UTC calendar days, excluding the entire target day. Use only data you are entitled to process.

| Column | Type | Definition / rule |
| --- | --- | --- |
| competition_id | string | `ENG_PL` or `COL_A`; constant, required by CLI |
| match_id | string | Unique ID beginning with competition plus `::` |
| date | ISO timestamp | Actual match timestamp, preferably explicitly UTC; converted to UTC |
| home | string | Team identity prefixed with competition plus `::` |
| away | string | Same namespace, different from home |
| home_npxg | float | Non-negative finite non-penalty expected goals created by home team |
| away_npxg | float | Corresponding away-team value |
| home_goals | integer | Non-negative official final home goals, including own goals and penalties |
| away_goals | integer | Corresponding away-team score |
| phase | optional string | If present, must be `regular_season` |

Artificial example only:

```csv
competition_id,match_id,date,home,away,home_npxg,away_npxg,home_goals,away_goals
ENG_PL,ENG_PL::demo1,2020-01-01T15:00:00Z,ENG_PL::A,ENG_PL::B,1.4,0.8,1,0
```

One row is insufficient to fit a model; the example illustrates schema only. The CLI rejects duplicate match IDs. Core validation rejects duplicate date/home/away keys, missing npxG, negative values, invalid goals, cross-league IDs and mixed competitions. Legacy low-level functions also accept untagged synthetic/Premier tables; use the CLI for strict public input boundaries.

## Definitions and source consistency

- npxG excludes penalty xG but includes non-penalty set pieces.
- The original Premier research uses published Understat team npxG, not an assumed sum of shot probabilities.
- The Colombian ingestion uses recorded penalty xG and internal reconciliation; a fixed universal penalty subtraction is not part of this public pipeline.
- Missing data must not be replaced by zero. A verified zero-shot match differs from an incomplete download.
- Historical provider revisions can still create availability bias even when match dates are correctly filtered.
- Raw provider data is not redistributed in this release. Record provider, definition, source URL, retrieval timestamp and transformations alongside any external dataset.

## Output fields

| Fields | Meaning |
| --- | --- |
| expected_npxg_home / away | Ratings-based expected non-penalty chance creation |
| goal_factor_home / away | Past weighted goals divided by past weighted npxG |
| lambda_home / away | Expected total goals after conversion |
| p_home / p_draw / p_away | Exhaustive outcome probabilities summing to one |
| p_over / p_under | Probabilities relative to 2.5 total goals in the replay |
| p_btts | Probability both teams score |
| outcome | Observed category: 0 home win, 1 draw, 2 away win |
| base_home / base_draw / base_away | Earlier-match historical outcome frequencies |
| new_team | At least one team uses the league-average fallback |

RPS in metrics uses outcome ordering home/draw/away and normalization by two. `n_eff` in fitted ratings is weighted exposure; `effective_sample_size` is a separate quantity, (sum of weights)^2 / sum of squared weights.
