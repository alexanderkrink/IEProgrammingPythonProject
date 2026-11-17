#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What drives a footballer's market value?
IE University - Programming for Data Management - Group Project (November 2026)

Authors: Alexander K., Áron F., Enzo G. and Krishiv S.

Pipeline: load + merge Transfermarkt tables -> clean + aggregate to one row per
player -> exploratory charts -> multiple linear regression with forward
stepwise feature selection.

Expected input (see data/README.md):
    data/players.csv
    data/appearances.csv

Run from anywhere:  python src/market_value_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Data lives in <repo>/data, resolved relative to this file so the script
# works on any machine without editing paths.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

players=pd.read_csv(DATA_DIR / "players.csv")
appearance=pd.read_csv(DATA_DIR / "appearances.csv")
combined = pd.merge(players, appearance, on="player_id", how="inner")
q75 = combined['minutes_played'].quantile(0.75)
q25 = combined['minutes_played'].quantile(0.25)

iqr_range = q75 - q25

import matplotlib
import matplotlib.pyplot as plt

plt.hist(combined['minutes_played'],bins=5, color="skyblue",edgecolor="black",alpha=0.7,rwidth=0.9)
plt.show() 

# =============================================================================
# Cleaning Data Set
# =============================================================================

combined.isna().sum()

#Fixing NaN values appeared in first and last names

# Extract the first word from 'name' where 'first_name' is missing
combined['first_name'] = combined['first_name'].fillna(combined['name'].str.split().str[0])
# Split 'name' into first and last name (split on the first space)
combined[['first_name', 'last_name']] = combined['name'].str.split(' ', n=1, expand=True)


# Extract everything after the first space from 'name' where 'last_name' is missing
combined['last_name'] = combined['last_name'].fillna(combined['name'].str.split(' ', n=1).str[1])


# Delete first and last name, player code, country and city_ob,  - its redundant
# Delete height, contract expiration, agent name, image url, url, game id, player club id, player current club id, date, player_name - no use


filtered_combined=combined.drop(columns=['first_name', 'last_name', 'player_code', 'country_of_birth', 'city_of_birth', 'height_in_cm', 'contract_expiration_date', 'agent_name','image_url', 'url', 'game_id', 'player_club_id', 'player_current_club_id', 'date', 'player_name'])

# Fixed code:
filtered_combined_test=filtered_combined.groupby('player_id', as_index=0).agg(appearances=('appearance_id', 'count'))

exploration=combined.loc[combined['sub_position'].isna(),  ['first_name', 'last_name', 'name']].nunique() #use this to check what values are nan



player_summary=filtered_combined.groupby(['player_id', 'name', 'last_season', 'current_club_id', 'country_of_citizenship', 'date_of_birth', 'sub_position', 'position', 'foot', 'international_caps', 'international_goals', 'current_national_team_id', 'current_club_name', 'market_value_in_eur', 'highest_market_value_in_eur', 'current_club_domestic_competition_id'], 
                                         as_index=False, 
                                         dropna=False).agg(total_appearances=('appearance_id', 'count'), yellow_cards=('yellow_cards', 'sum'), red_cards=('red_cards', 'sum'), goals=('goals', 'sum'), assists=('assists', 'sum'), minutes_played=('minutes_played', 'sum'))

filtered_player_summary=player_summary.loc[player_summary['current_club_domestic_competition_id'].isin(['GB1', 'L1', 'ES1', 'FR1', 'IT1'])]

      
# =============================================================================
# Data manipulation
# =============================================================================

#Insight 1
#Correlation between key drivers

import seaborn as sns

plt.figure(figsize=(10, 10))
corr_cols = ['market_value_in_eur', 'total_appearances', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards']
corr_matrix = filtered_player_summary[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title(' Correlation Heatmap (Drivers of Market Value)', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

#Insight 2
#Position Valuation 
#Filter out rows where position is Missing

boxplot_data = filtered_player_summary[~filtered_player_summary['position'].isin(['Missing'])].dropna(subset=['position'])
#we used an LLM to help us figure out how to sort out the missing value from position


#With outliers
plt.figure(figsize=(10, 10))
sns.boxplot(x='position', y=boxplot_data['highest_market_value_in_eur']/1000000, data=boxplot_data, palette='Pastel1', showmeans=True, meanprops={"marker":"o", "markerfacecolor":"red", "markersize":"8"})
plt.title('Valuation Distribution by Position', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Position', fontsize=12, fontweight='bold')
plt.ylabel('Peak Market Value (Millions of Euro)', fontsize=12, fontweight='bold')
plt.plot([], [], marker='o', color='red', linestyle='None', label='Mean for Given Position')
plt.legend(loc='upper right')
plt.show()


#Without outliers
q1=filtered_player_summary['highest_market_value_in_eur'].quantile(0.25)
q3=filtered_player_summary['highest_market_value_in_eur'].quantile(0.75)

iqr=q3 - q1

lower_bound= q1 - 1.5 * iqr
upper_bound= q3 + 1.5 * iqr
non_outliers_df = boxplot_data[(filtered_player_summary['highest_market_value_in_eur'] >= lower_bound) & (filtered_player_summary['highest_market_value_in_eur'] <= upper_bound)]

plt.figure(figsize=(10, 10))
sns.boxplot(x='position', y=non_outliers_df['highest_market_value_in_eur']/1000000, data=non_outliers_df, palette='Pastel1', showmeans=True, meanprops={"marker":"o", "markerfacecolor":"red", "markersize":"8"})
plt.title('Valuation Distribution by Position - Without Outliers', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Position', fontsize=12, fontweight='bold')
plt.ylabel('Peak Market Value (Millions of Euro)', fontsize=12, fontweight='bold')
plt.plot([], [], marker='o', color='red', linestyle='None', label='Mean for Given Position')
plt.legend(loc='upper right')
plt.show()


#Insight 3
#Goal contributions to valuation

filtered_player_summary['goal_contributions'] = filtered_player_summary['goals'] + filtered_player_summary['assists']
goal_contributions_data = filtered_player_summary[~filtered_player_summary['position'].isin(['Missing'])].dropna(subset=['position'])
plt.figure(figsize=(10, 10))
sns.scatterplot(data=goal_contributions_data, x='goal_contributions', y=goal_contributions_data['highest_market_value_in_eur']/1000000,  hue='position',  sizes=(50, 300),palette='Set2',size='goals')

plt.title('Goal Contributions vs Peak Market Value', fontsize=14, fontweight='bold')
plt.xlabel('Goal Contributions', fontsize=12, fontweight='bold')
plt.ylabel('Peak Market Value (Millions of Euro)', fontsize=12, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# =============================================================================
# Regression Model
# =============================================================================

# 1. Feature Definition
num_cols = ['total_appearances', 'yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played', 'age']
cat_cols = ['position', 'foot', 'current_club_domestic_competition_id']  # 'domestic_league_code' or 'league' column
target = 'market_value_in_eur'

# 2. Compute Age & Preprocess Data
# Convert date of birth to datetime and compute age relative to last season / current date
if 'date_of_birth' in filtered_player_summary.columns:
    filtered_player_summary['date_of_birth'] = pd.to_datetime(filtered_player_summary['date_of_birth'], errors='coerce')
    # Calculating age (using last_season year or reference year)
    reference_year = pd.to_datetime('2024-01-01') # or use filtered_player_summary['last_season']
    filtered_player_summary['age'] = (reference_year - filtered_player_summary['date_of_birth']).dt.days // 365.25

# Subset dataframe to selected columns
df = filtered_player_summary[num_cols + cat_cols + [target]].dropna().copy()

# Ensure numerical columns are properly formatted as floats/ints
for col in num_cols + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# One-hot encode categorical variables (position, foot, league)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# 3. Exploratory Scatter Plots (Including Age)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(df_encoded['age'], df_encoded[target], color='#2ca02c')      
axes[0, 0].set_xlabel('age')                                   
axes[0, 0].set_ylabel(target)
axes[0, 0].set_title(f'{target} vs age')

axes[0, 1].scatter(df_encoded['goals'], df_encoded[target], c='#1f77b4')
axes[0, 1].set_xlabel('goals')
axes[0, 1].set_ylabel(target)
axes[0, 1].set_title(f'{target} vs goals')

axes[1, 0].scatter(df_encoded['assists'], df_encoded[target], c='#ff7f0e')
axes[1, 0].set_xlabel('assists')
axes[1, 0].set_ylabel(target)
axes[1, 0].set_title(f'{target} vs assists')

axes[1, 1].scatter(df_encoded['minutes_played'], df_encoded[target], color='#d62728')
axes[1, 1].set_xlabel('minutes_played')
axes[1, 1].set_ylabel(target)
axes[1, 1].set_title(f'{target} vs minutes_played')

plt.tight_layout()
plt.show()

# 4. Train/Test Split
data_train, data_test = train_test_split(df_encoded, test_size=0.20, random_state=42)

features = [col for col in df_encoded.columns if col != target]

X_train = data_train[features]
y_train = data_train[target]

X_test = data_test[features]
y_test = data_test[target]

# 5. Initial Model Fit & R^2
mvmodel = LinearRegression()
mvmodel.fit(X_train, y_train)

r_squared = mvmodel.score(X_train, y_train)
print(f"coefficient of determination (R^2): {r_squared}")

# 6. Residual Plot
error = y_train - mvmodel.predict(X_train)
plt.figure(figsize=(5, 4))
plt.scatter(mvmodel.predict(X_train), error, color='blue')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('fitted value')
plt.ylabel('error')
plt.show()

# 7. Forward Stepwise Feature Selection
model = LinearRegression()
# Set k_features to 5 or 6 since domestic league code expands into multiple dummy variables
forward_stepwise = sfs(model, k_features=5, forward=True, verbose=2, scoring="r2")    
forward_stepwise = forward_stepwise.fit(X_train, y_train)

print(f"\nTop features selected: {forward_stepwise.k_feature_names_}\n")

# 8. Final Model Coefficients & Intercept
selected_features = list(forward_stepwise.k_feature_names_)
model.fit(X_train[selected_features], y_train)

coefficients = pd.Series(model.coef_, index=selected_features)
intercept = model.intercept_

print(coefficients)
print(f"Intercept is {intercept}")

#=============================================================================
# MODEL RESULTS INTERPRETATION
# =============================================================================
#
# 1. OVERALL MODEL PERFORMANCE (R^2 = 0.2545 / 25.45%)
# -----------------------------------------------------------------------------
# - The initial full model achieved an R^2 of 0.2545, meaning the selected 
#   predictors account for approximately 25.45% of the total variation in player 
#   market values.
# - The forward stepwise selection built up performance across 5 steps:
#     * Feature 1: score = 0.1071 (captures ~10.7% of variance on its own)
#     * Feature 2: score = 0.2129 (+10.6% boost by adding age/league)
#     * Feature 3: score = 0.2332 (+2.0% gain)
#     * Feature 4: score = 0.2415 (+0.8% gain)
#     * Feature 5: score = 0.2418 (final selected model score ~24.18%)
#
# 2. FEATURE COEFFICIENTS INTERPRETATION
# -----------------------------------------------------------------------------
# - Premier League Flag ('current_club_domestic_competition_id_GB1': +€4,161,503.00)
#   Playing in the English Premier League is the single largest positive driver. 
#   Holding all else equal, Premier League players command a baseline premium 
#   of ~€4.16M compared to players in reference leagues.
#
# - Age ('age': -€563,788.40)
#   Holding all performance stats constant, each additional year of age reduces a 
#   player's valuation by ~€563.8k, reflecting the market premium placed on 
#   youth and long-term resale potential.
#
# - Assists ('assists': +€225,033.70)
#   Direct goal creation heavily increases valuation. Holding all other factors 
#   equal, each additional assist adds ~€225k to a player's valuation.
#
# - Minutes Played ('minutes_played': +€230.25)
#   Volume of play provides steady valuation growth. Each minute played adds 
#   €230.25, which equals approximately €20,722 per full 90-minute appearance.
#
# - Yellow Cards ('yellow_cards': -€23,310.81)
#   Disciplinaries carry a minor valuation penalty: each yellow card reduces a 
#   player's estimated value by ~€23.3k (reflecting risk of suspension).
#
# 3. BASELINE INTERCEPT (€17,364,116.28)
# -----------------------------------------------------------------------------
# - The intercept represents the mathematical constant baseline valuation 
#   before subtracting for age and adding adjustments for playing stats and league.
#
# 4. FITTED REGRESSION EQUATION
# -----------------------------------------------------------------------------
# Market Value (€) = 17,364,116.28 + 4,161,503.00*(Premier_League) 
#                    + 225,033.70*(Assists) + 230.25*(Minutes_Played) 
#                    - 563,788.40*(Age) - 23,310.81*(Yellow_Cards)
# =============================================================================



#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set clean aesthetic style for presentation graphics
sns.set_theme(style="whitegrid")

# Preprocessing data for Graph 2 & Graph 3
filtered_player_summary['age_group'] = pd.cut(
    filtered_player_summary['age'], 
    bins=[0, 22, 28, 33, 100], 
    labels=['Under 23\n(Prospects)', '23-28\n(Peak)', '29-33\n(Veterans)', '34+\n(Late Career)']
)

filtered_player_summary['goal_contributions'] = (
    filtered_player_summary['goals'] + filtered_player_summary['assists']
)


# =============================================================================
# GRAPH 1: Baseline Valuation by Domestic League
# =============================================================================
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=filtered_player_summary, 
    x='current_club_domestic_competition_id', 
    y='market_value_in_eur', 
    palette='Set2'
)
plt.title('Baseline Market Value by League Tier', fontsize=12, fontweight='bold')
plt.xlabel('League Code', fontsize=10)
plt.ylabel('Market Value in EUR (€)', fontsize=10)
plt.yscale('log')  # Log scale handles valuation outliers clearly
plt.tight_layout()
plt.show()


# =============================================================================
# GRAPH 2: The effect on valuation of a punishment card per position
# =============================================================================
sns.set_theme(style="whitegrid")

# 1. Filter data for regular players (at least 450 minutes / 5 matches) and main positions
positions = ['Defender', 'Midfield', 'Attack']
df_filtered = filtered_player_summary[
    (filtered_player_summary['position'].isin(positions)) & 
    (filtered_player_summary['minutes_played'] >= 450)
].copy()

# 2. Normalize card rates per 1,000 minutes played
df_filtered['yellows_per_1k_min'] = (df_filtered['yellow_cards'] / df_filtered['minutes_played']) * 1000
df_filtered['reds_per_1k_min'] = (df_filtered['red_cards'] / df_filtered['minutes_played']) * 1000

# 3. Create 2x3 Grid Dashboard
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)

yellow_color = '#e6a100'
red_color = '#d9534f'

for idx, pos in enumerate(positions):
    pos_data = df_filtered[df_filtered['position'] == pos].dropna(subset=['market_value_in_eur'])
    
    # -------------------------------------------------------------------------
    # TOP ROW: Yellow Cards per 1k Mins
    # -------------------------------------------------------------------------
    clean_yellow = pos_data.dropna(subset=['yellows_per_1k_min'])
    
    # Calculate slope (slope = change in market value (€) per 1 yellow card / 1,000 mins)
    slope_y, _ = np.polyfit(clean_yellow['yellows_per_1k_min'], clean_yellow['market_value_in_eur'], 1)
    
    sns.regplot(
        data=clean_yellow, 
        x='yellows_per_1k_min', 
        y='market_value_in_eur',
        ax=axes[0, idx],
        scatter_kws={'alpha': 0.4, 'color': yellow_color},
        line_kws={'color': 'black', 'linewidth': 2},
        label=f'Slope: €{slope_y:,.0f} / card'
    )
    axes[0, idx].set_title(f'{pos}: Yellow Cards', fontsize=12, fontweight='bold')
    axes[0, idx].set_xlabel('Yellow Cards per 1,000 Mins', fontsize=10)
    axes[0, idx].legend(loc='upper right', frameon=True)

    # -------------------------------------------------------------------------
    # BOTTOM ROW: Red Cards per 1k Mins
    # -------------------------------------------------------------------------
    clean_red = pos_data.dropna(subset=['reds_per_1k_min'])
    
    # Calculate slope (slope = change in market value (€) per 1 red card / 1,000 mins)
    slope_r, _ = np.polyfit(clean_red['reds_per_1k_min'], clean_red['market_value_in_eur'], 1)
    
    sns.regplot(
        data=clean_red, 
        x='reds_per_1k_min', 
        y='market_value_in_eur',
        ax=axes[1, idx],
        scatter_kws={'alpha': 0.4, 'color': red_color},
        line_kws={'color': 'black', 'linewidth': 2},
        label=f'Slope: €{slope_r:,.0f} / card'
    )
    axes[1, idx].set_title(f'{pos}: Red Cards', fontsize=12, fontweight='bold')
    axes[1, idx].set_xlabel('Red Cards per 1,000 Mins', fontsize=10)
    axes[1, idx].legend(loc='upper right', frameon=True)

# Set common Y-axis labels for leftmost column
axes[0, 0].set_ylabel('Market Value in EUR (€)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Market Value in EUR (€)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# GRAPH 3: Nationality Breakdown of Top 100 Most Valuable Players
# =============================================================================
sns.set_theme(style="whitegrid")

# 1. Select the top 100 most valuable players
top_100_players = (
    filtered_player_summary
    .dropna(subset=['market_value_in_eur', 'country_of_citizenship'])
    .sort_values(by='market_value_in_eur', ascending=False)
    .head(100)
)

# 2. Count player occurrences by nationality
nationality_counts = top_100_players['country_of_citizenship'].value_counts()

# 3. Take the top 14 nationalities and group the rest into 'Other'
top_n = 14
main_nations = nationality_counts.head(top_n)
other_count = nationality_counts.iloc[top_n:].sum()

if other_count > 0:
    pie_data = pd.concat([main_nations, pd.Series({'Other': other_count})])
else:
    pie_data = main_nations

# 4. Create the Pie Chart
plt.figure(figsize=(10, 8))

# Palette expanded to accommodate 14+ distinct slices
colors = sns.color_palette('tab20', len(pie_data))

plt.pie(
    pie_data.values, 
    labels=pie_data.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=colors,
    pctdistance=0.82,  # Places percentages nicely inside/near slices
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    textprops={'fontsize': 9.5}
)

plt.title('Nationality Breakdown of Top 100 Most Valuable Players', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

















