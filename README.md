# What Drives a Footballer's Market Value?

*An analysis of player valuations in Europe's top five leagues.*

Group project for **Programming for Data Management** (IE University), November 2026.

**Authors:** Alexander K., Áron F., Enzo G. and Krishiv S.

## Overview

Transfer fees are among the largest financial decisions a football club makes, yet a
player's "market value" is an estimate, not a calculated number. This project asks
**which measurable factors explain a footballer's market value**, using two
[Transfermarkt](https://github.com/dcaribou/transfermarkt-datasets) tables (player
profiles and match appearances) for players at clubs in the Premier League, La Liga,
Bundesliga, Serie A and Ligue 1.

The script merges and cleans the data, aggregates it to one row per player, explores it
with six visualisations, and fits a multiple linear regression with forward stepwise
feature selection.

The full write-up (problem, data, methods, results, limitations) is in
[`report/Programming_Report.pdf`](report/Programming_Report.pdf).

## Research questions

1. Which performance statistics are most strongly related to market value?
2. Does a player's position change how they are valued?
3. Do goal contributions (goals + assists) translate into a higher peak value?
4. Is poor discipline (yellow/red cards) penalised, and does this differ by position?
5. Which nationalities dominate the top end of the market?
6. Can these factors be combined into a model that estimates value, and how well does it perform?

## Key findings

- **No single statistic dominates.** Assists (r = 0.32) and goals (r = 0.30) correlate most with market value; appearances and minutes sit around 0.26-0.27.
- **League and age matter more than contributions.** Playing in the Premier League adds about €4.16M; each extra year of age removes about €0.56M (all else equal).
- **Position sets the ceiling, not the average.** Outfield medians are similar, but the most extreme valuations belong to attackers and midfielders; goalkeepers are valued lowest.
- **Discipline is penalised once time-adjusted.** Raw card counts correlate positively with value, but cards per 1,000 minutes correlate negatively.
- **Elite value is concentrated.** England and France supply a third of the 100 most valuable players.
- **About 75% of value remains unexplained** (R² ≈ 0.25), pointing to reputation, potential and contract situation, which the data cannot capture.

Fitted model (5 features selected by forward stepwise regression):

```
Value (€) = 17,364,116 + 4,161,503 × PremierLeague + 225,034 × Assists
            + 230.25 × MinutesPlayed − 563,788 × Age − 23,311 × YellowCards
```

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md              # data dictionary and sources
│   ├── players.csv            # player profiles (Transfermarkt)
│   └── appearances.csv        # per-match statistics (Transfermarkt)
├── src/
│   └── market_value_analysis.py   # full pipeline: load → clean → explore → model
└── report/
    └── Programming_Report.pdf     # written report
```

## How to run

**Requirements:** Python 3.9+

```bash
git clone https://github.com/alexanderkrink/IEProgrammingPythonProject.git
cd IEProgrammingPythonProject
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/market_value_analysis.py
```

Make sure `data/players.csv` and `data/appearances.csv` are present (see
[`data/README.md`](data/README.md)). The script resolves data paths relative to the
repository, so no path editing is needed. It can also be run top to bottom in Spyder or
PyCharm.

The charts open one after another (close each window to advance), and the regression
results (R², selected features, coefficients, intercept) print to the console.

## What the script does

1. **Load and merge** `players.csv` and `appearances.csv` on `player_id` (inner join).
2. **Clean:** rebuild missing first/last names from the full name, drop 15 redundant or unused columns.
3. **Aggregate** match rows into one row per player (appearances, goals, assists, cards, minutes) and keep players whose club is in GB1, L1, ES1, FR1 or IT1.
4. **Explore:** correlation heatmap, valuation by position (with and without outliers), goal contributions vs peak value, valuation by league, a yellow/red card dashboard per position, and nationality of the top 100 players.
5. **Model:** one-hot encode categoricals, 80/20 train/test split, linear regression, forward stepwise feature selection (`mlxtend`), then print coefficients.

## Tools

pandas · NumPy · Matplotlib · seaborn · scikit-learn · mlxtend

## Limitations

Statistics are career totals while market value is a snapshot of today; no minimum-minutes filter is applied; the R² is reported on training data (the test set is not yet scored); and results show association, not causation. See report section 9 for details and proposed extensions.
