# msis-assignments
MSIS Machine Learning Algorithm Course

# The Price-Tag Decision: Logistic Regression and Menu Costs at a Large Retail Store

MSIS Graduate Assignment — Florida International University, MS in Information Systems
Instructor: Dr. Hemang Subramanian

**Authors:** Luis Paredes, Javier Brito Martillo

## Business scenario

SunCoast Retail Mart stocks 1,500 SKUs. Every pricing cycle, the store has to decide whether
to change a price tag. Repricing corrects competitive misalignment and restores margin after
supplier cost changes, but every tag change has a menu cost — the labor and materials of
relabeling. About 30% of shelves use electronic shelf labels (ESL), where repricing is nearly
free; the rest use printed paper tags that cost real money to change.

This project models the relabeling decision with logistic regression, using one cycle of data
(`retail_relabel.csv`, 1,500 SKUs) that records whether each tag was changed, along with
competitor prices, unit cost, cost changes, sales velocity, and relabel cost. The goal is to turn
the model into a repricing policy the store owner can act on.

## What's in this repo

| File | Description |
|---|---|
| `Business_Problem_About_Retail_Relabeling_using_Logistic_Regression.ipynb` | Full analysis notebook — feature engineering, both models, out-of-sample evaluation, economic analysis, and written interpretations |
| `retail_relabel.csv` | Source data: 1,500 SKUs with competitor prices, costs, sales velocity, ESL flag, and the relabel outcome |
| `Retail_Relabel_Writeup_Brito_Paredes.pdf` | Write-up covering Parts 1–5, the memo to the store owner, and a discussion-questions section |

## What we did

**Part 1 — Feature engineering & EDA.** Built `CompAvg` (average of three competitor prices),
the signed gap `GapPct`, its absolute value `AbsGapPct`, `LogUnits` (log of weekly units sold),
and `MarginPct`. The overall relabel base rate is 40%. The most important early finding: items
on electronic shelf labels get relabeled at 54.1% versus 33.9% for paper tags — direct evidence
that menu cost, not just misalignment, drives whether a mispriced item ever gets fixed.

**Part 2 — Naive model (Model 1).** Fit `Relabel ~ GapPct + UnitCost + RelabelCost` using the
*signed* gap. We show deductively (and confirm empirically) why this is the wrong functional
form: the owner's decision to relabel is symmetric around zero (over- and under-priced items
both get corrected), but a signed coefficient can only fit one direction of that relationship.

**Part 3 — Economically motivated model (Model 2).** Fit
`Relabel ~ AbsGapPct + CostChangePct + LogUnits + RelabelCost + DaysSinceLastChange + UnitCost`.
Switching to the absolute gap alone raises pseudo-R² from 0.077 to 0.22; adding the full set of
economically motivated predictors takes it to 0.275, with AIC dropping from 1,871 to 1,478.
`CostChangePct` is highly significant while the *level* of `UnitCost` is not — exactly what
menu-cost theory predicts, since a stale tag is caused by a recent change in conditions, not by
how expensive an item is.

**Part 4 — Out-of-sample evaluation.** A 70/30 stratified train/test split (`random_state=7`)
gives 77.6% accuracy and 0.853 AUC on held-out data, confirming Model 2 generalizes rather than
just fitting the training sample.

**Part 5 — Economic decision analysis & memo.** For each test SKU we computed
`ExpectedWeeklyLoss` and `NetBenefit` against the relabel cost. 98.7% of test SKUs would recoup
their relabel cost within a single week of corrected pricing. The accompanying memo to the store
owner covers which SKU segments to relabel first, the quantified business case for converting
more shelves to ESL, and an explicit caution that the results are observational (associational,
not causal).

## How to run

pip install numpy pandas statsmodels scikit-learn
jupyter notebook Business_Problem_About_Retail_Relabeling_using_Logistic_Regression.ipynb

The notebook expects `retail_relabel.csv` in the same directory and runs end to end.

## Academic integrity note

AI assistants were used for coding support and formatting. All interpretation, functional-form
reasoning, and the memo reflect our own analysis and can be defended orally.
