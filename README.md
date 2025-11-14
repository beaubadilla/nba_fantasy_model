# NBA Fantasy Model
Regression model to predict a player's fantasy average for the upcoming season. \
Input: Player's previous year's stats
Output: Fantasy Point Average per Game for next season

## Run Tests
`pytest tests/`
`make tests`

## `Make` commands
Check Makefile for extensive list of commands
`make test`
`make format`
`make run`

## Run the data pipeline
The end-to-end preprocessing pipeline expects a root folder of season subdirectories and writes both the imputed dataset and engineered-feature export.

```bash
make run
```

The command above reads all seasons inside `data/raw/`, saves the imputed dataset to `data/interim/imputed.csv`, and writes the feature-enhanced version to the same directory with the `_features.csv` suffix.

## Fantasy Points (FP) Calculation
1 point = 1.0 FP \
1 assist = 1.5 FP \
1 rebound = 1.2 FP \
1 block = 2.0 FP \
1 steal = 2.0 FP \
1 turnover = -1.0 FP

## Standard 9 Categories
If we ever want to build a model for a Cat9 league
1. Points
2. Rebounds
3. Assists
4. Steals
5. Blocks
6. Three-Pointers
7. Field Goal Percentage (FG%)
8. Free Throw Percentage (FT%)
9. Turnovers

## Row
Next Season = 22-23  
All stats are for regular season only  
```
Player Name
* [TARGET] FP per game next season (22-23)
* Points per game in previous season (21-22)
* Assists per game in previous season (21-22)
* Rebounds per game in previous season (21-22)
* Blocks per game in previous season (21-22)
* Steals per game in previous season (21-22)
* Turnovers per game in previous season (21-22)
* Points per game for career up until 21-22 inclusive
* Assists per game for career until 21-22 inclusive
* Rebounds per game for career until 21-22 inclusive
* Blocks per game for career until 21-22 inclusive
* Steals per game for career until 21-22 inclusive
* Turnovers per game for career until 21-22 inclusive
[DONE] 1. Is 21-22 coach the same as the upcoming 22-23 coach?
[DONE] 2. Is this player's team the same between 21-22 and the upcoming 22-23 season?
Number of new players on team with high minutes
[DONE] Age
[DONE] 3. Number of years in NBA until 21-22 inclusive
Usage rate/number X option
Total usage rate of roster
is hungry/motivated (contract year, fiba/olympic year, just on fiba/olympic, deep playoff run)
number of all star teammates
number of all nba teammates
number of awards
4. current team's offensive rating last year
5. current team's defensive rating last
[DONE] Position
Number of players within the team with the same position
Number of injured players
Height
```
---
Curry
23_24
fp for 23_24 [target]
[x] team
[x] pts for 23_24
[x] rebs for 23_24
[x] asts for 23_24
[v] pts for 22_23
[v] rebs for 22_23
[v] ast for 22_23
[v] running pts until 23_24
[v] running rebs until 23_24
[v] running ast until 23_24
[v] does Curry have a new coach for 23_24? 
    - compare 22_23 with 23_24
    - assume we have data of 23_24 coaches
        * to create the training dataset, we use raw data
        * in the future, to run inference, we need to have a separate dataset for the upcoming season
[v] did Curry move teams for 23_24?
    - compare 22_23 with 23_24
    - assume we have data of 23_24 roster
        * to create the training dataset, we use raw data
        * in the future, to run inference, we need to have a separate dataset for the upcoming season


-------------
[x] Player-additional
[x] season_start
[x] Rk
Age
[x] Team
G
[x] GS (Games Started)
[x] MP
[x] FG
[x] FGA
[x] FG%
[x] 3P
[x] 3PA
[x] 3P%
[x] 2P
[x] 2PA
[x] 2P%
[x] eFG%
[x] FT
FTA
[x] FT%
[x] ORB
[x] DRB
TRB (season rpg)
AST (season apg)
STL (season spg)
BLK (season bpg)
TOV (season tpg)
[x] PF (season pfpg)
PTS (season ppg)
[x] Awards
[x] did_play
[x] Player
Pos
years_in_nba
[x] total_PTS (season total points)
[x] total_AST
[x] total_TRB
[x] total_BLK
[x] total_STL
[x] total_TOV
career_PTS_pg (running career points per game)
career_AST_pg
career_TRB_pg
career_BLK_pg
career_STL_pg
career_TOV_pg
new_coach
new_team

# Case Studies
Rookie

# TODO
* Find players with significant 1 season spike
* Find players with significant 1 season drop
* Find players who fluctuate
* Find players who rose to high FP from low FP
* Find players who fell to low FP from high FP
* Are rookies worth risking for?
* Do tanking teams have high FP players?

# Questions
[DONE] * Should we filter out rows by games (G) [played]? Let's try it
[DONE] * Should we remove rookie seasons? Yes. Create a separate model for rookie seasons
* For players that played multiple teams within a season, do we use their total, first team, last team, or X team for singular season stats? Last team. When I think about drafting players at the beginning of the season, I don't care that much how well they did in any team except their most recent one (which is often the same team for the next.)

# Notes
We should do all filtering once the dataset is complete. This is in regards to merging, and then calculating, career stats. Initially, I filtered out games at the beginning of data prep. However, when I started creating cumulative features, it was inaccurate because rows were missing.
