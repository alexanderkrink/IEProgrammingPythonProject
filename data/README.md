# Data

The analysis uses two tables from the open-source
[Transfermarkt football dataset](https://github.com/dcaribou/transfermarkt-datasets).

| File | One row represents | Key columns used |
|------|--------------------|------------------|
| `players.csv` | A player's profile and current status | `player_id`, `name`, `date_of_birth`, `position`, `sub_position`, `foot`, `country_of_citizenship`, `current_club_domestic_competition_id`, `market_value_in_eur`, `highest_market_value_in_eur` |
| `appearances.csv` | One player's performance in one match | `appearance_id`, `player_id`, `goals`, `assists`, `yellow_cards`, `red_cards`, `minutes_played` |

The two tables are joined on `player_id` (inner join), then aggregated to one row
per player. See the [report](../report/Programming_Report.pdf), sections 3 and 4.

Both files must sit in this folder with exactly these names:

```
data/players.csv
data/appearances.csv
```
