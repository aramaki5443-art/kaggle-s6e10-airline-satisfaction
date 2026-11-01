# Kaggle Playground S6E10 — Predicting Airline Satisfaction

Binary classification: predict whether an airline passenger is satisfied from a survey (age, class, travel type, flight distance, delays, and 14 service ratings scored 0–5). Metric: **ROC AUC**. Data: 699,635 training rows and 299,844 test rows of synthetic data generated from a real airline survey.

日本語版は [README.ja.md](README.ja.md) をご覧ください。

## Result

**Final: Private LB 0.96239, rank 52 / 3,748 teams (top 1.4%).** The Public rank was 513, so the final position rose by 461 places.

| Submission | Private LB | Public LB |
|---|---|---|
| **Best stack by CV (226 members + shallow LightGBM blend), selected** | **0.96239** | 0.96182 |
| Hedge: 50/50 rank average with a public LB–tuned file, selected | 0.96236 | 0.96202 |
| Own models only (5-model stack) | 0.96208 | 0.96123 |
| Baseline (LightGBM, raw 20 columns) | 0.95910 | 0.95817 |

The submission chosen by cross-validation was my best Private score among all submissions. The hedge scored higher on the Public LB but lower on the Private LB. Most of the jump in rank came from many teams that had submitted public LB–tuned files dropping on the Private LB: 71% of the Public top 360 finished below rank 500. My own Private−Public gain (+0.00034) was close to the median of teams with similar Public ranks (+0.00032).

![Public LB progress](docs/img/lb_progress.png)

## Approach

| Phase | What I did | Main outcome (OOF AUC) |
|---|---|---|
| EDA | Univariate AUC per feature, cross tabs, the meaning of "0" ratings, per-segment heatmaps | Found that a rating's effect depends on the customer segment (e.g. comfort matters only for loyal business travelers) |
| Feature engineering | LightGBM, adding one feature at a time and judging over 3 CV seeds | Raw 20 columns 0.95846 → with engineered features 0.96046 (default parameters) |
| Models | LightGBM, XGBoost, CatBoost, RealMLP, TabPFN | Single models 0.9608–0.9611 |
| Hyperparameters | Optuna (LightGBM, XGBoost, CatBoost on Kaggle GPU) | LightGBM +0.00039 alone, but only +0.00005 in the blend |
| Ensemble | Stacking with logistic regression on logits, evaluated on a separate fold split | 5-model stack 0.961685 |
| Public OOF | Added other participants' published OOF predictions as stack members | 226 members + shallow LightGBM blend 0.962266 |

![Univariate AUC](docs/img/univariate_auc.png)

### What worked

| Idea | Gain (OOF AUC) | Why I think it worked |
|---|---|---|
| Target encoding of the exact Flight Distance value | +0.00136 | Only 3,474 distinct distances; each behaves like a route ID with its own satisfaction rate. Trees would need thousands of splits to learn this |
| "Gate" features: a rating kept only for the segment where it matters (others set to missing) | +0.00005 to +0.00009 each | From the EDA: comfort ratings predict satisfaction for business travelers (AUC 0.72–0.79) but not for personal travel (≈0.50). The top SHAP feature was one of these |
| Expected value of a rating predicted from the other 20 columns (no labels used) | +0.00004 to +0.00009 | A passenger who rates lower than expected is less satisfied |
| Route profile (mean rating per distance value) and segment deviations | +0.00004 to +0.00006 | Summaries a tree cannot build in one split |
| Adding diverse models (RealMLP, TabPFN) to tree models | +0.001 (LB) | Rank correlation with the trees was 0.96, compared with 0.98–0.99 between trees |
| Stacking with published OOF predictions | +0.00054 (LB +0.0006) | More model families (tabular foundation models, TabPFN variants) than I could train myself |

![SHAP of the LightGBM model](docs/img/shap_lightgbm.png)

### What did not work (selected)

- Adding the original real dataset as extra training rows: −0.00017. Using a model trained on it as a single "teacher" feature helped OOF (+0.00038) but lowered the Public LB by about 0.0002, so I left it out.
- Digit features, GPT-2 token keys, and interaction products: no gain on top of target encoding.
- TabM (another neural net): its predictions differed the most from the others (rank correlation 0.94), but its single score was 0.0024 lower, so it got no weight. Being different is necessary but not sufficient.
- Dropping stack members with negative coefficients: −0.00015. A negative coefficient offsets similar members and does not mean the member is useless.

## Validation

- 5-fold StratifiedKFold (seed 42). Every comparison uses the same folds and rows.
- **Adopt a feature only if the mean gain over 3 CV seeds is at least +0.00003 and at least 2 of 3 seeds improve.** Changing only the fold seed moved the difference by about 0.00005, so a single-seed result was not trustworthy.
- Stacking was evaluated on a separate 5-fold split (seed 7): the blend is learned on 4 folds and scored on the remaining one.
- Public LB (about 60k rows) cannot distinguish differences below about 0.0003, so decisions were based on OOF (700k rows).

## My work vs. external resources

- **Implemented myself:** EDA, all feature engineering code, the 3-seed adoption framework, LightGBM/XGBoost/CatBoost pipelines and tuning, stacking and its evaluation. The code is in `src/` and `experiments/`.
- **Ideas taken from public notebooks and re-implemented in my own code:** original-data teacher, expected-value features, route profile, segment statistics, all-categorical CatBoost.
- **Reproduced from public notebooks (code not included in this repository):** the RealMLP configuration (yekenot) and the TabPFN memory workaround (hemingweb).
- **Used as-is (not included):** published OOF predictions from about 20 Kaggle datasets (cdeotte and others) as stack members in the final submission.
- **Final submission:** slot 1 is the best stack by CV (Private 0.96239). Slot 2 is a hedge: a 50/50 rank average of slot 1 and a public LB–tuned public submission (Private 0.96236). Slot 2 has no CV backing and did slightly worse on the Private LB.

## Lessons

1. In synthetic data, check value-level patterns first. The exact-value target encoding was the largest single gain.
2. EDA findings such as segment-dependent effects translate directly into features.
3. For ensembles, add models that are both different and strong. Gains from hyperparameter tuning shrank to 1/5–1/10 once the models were blended.
4. At the top of the leaderboard, the gap came from the number and variety of models. Check early whether public OOF predictions exist.
5. Confirm small differences over several seeds and do not chase small Public LB differences. Here, trusting CV moved the final rank from 513 (Public) to 52 (Private).

## Repository layout

```
src/            lgb_fe_check.py (features + LightGBM + 3-seed check), aux_expected_value.py,
                xgb_check.py, catboost_kaggle.py (Kaggle GPU), stack_check.py (stacking evaluation)
experiments/    lgb_forward_add.py, lgb_optuna.py, shap_report.py
notebooks/      EDA.ipynb (with outputs)
docs/           review_ja.pdf (Japanese slide deck, 51 slides), img/
```

## Reproduce

```bash
uv sync
# put train.csv and test.csv from the competition page into data/
python src/lgb_fe_check.py --help          # each script's docstring explains its purpose and options
python src/aux_expected_value.py --col "Online boarding" --name boarding
```

Scripts are run from the repository root. Comments and docstrings are in Japanese.

## Development note

I carried out the analysis and implementation with the help of an AI coding assistant (Claude Code). I made the decisions on which experiments to run and which results to adopt, and recorded them in my own experiment log.
