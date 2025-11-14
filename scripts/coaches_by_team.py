import csv
import glob
import os
from collections import defaultdict


OUTPUT_DIR = os.path.join('data', 'interim')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect coaches per team-season while tracking games coached for tie-breaking.
team_data = defaultdict(lambda: defaultdict(list))
for path in sorted(glob.glob(os.path.join('data', 'raw', '*', 'coaches.csv'))):
    season = os.path.basename(os.path.dirname(path))
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            continue
        try:
            g_index = header.index('G')
        except ValueError:
            g_index = 2  # fall back to the expected position after 'Coach' and 'Tm'
        for row in reader:
            if not row:
                continue
            coach = row[0].strip()
            tm = row[1].strip()
            games = 0
            if len(row) > g_index:
                try:
                    games = int(row[g_index])
                except ValueError:
                    games = 0
            entries = team_data[tm][season]
            if not any(entry['coach'] == coach for entry in entries):
                entries.append({'coach': coach, 'games': games})

summary_path = os.path.join(OUTPUT_DIR, 'coaches_by_team_summary.csv')
first_path = os.path.join(OUTPUT_DIR, 'coaches_by_team_first_only.csv')
max_games_path = os.path.join(OUTPUT_DIR, 'coaches_by_team_max_games.csv')

with open(summary_path, 'w', newline='') as summary_file, \
        open(first_path, 'w', newline='') as first_file, \
        open(max_games_path, 'w', newline='') as max_games_file:
    summary_writer = csv.writer(summary_file)
    first_writer = csv.writer(first_file)
    max_games_writer = csv.writer(max_games_file)

    summary_writer.writerow(['Tm', 'Season', 'Coaches'])
    first_writer.writerow(['Tm', 'Season', 'Coach'])
    max_games_writer.writerow(['Tm', 'Season', 'Coach', 'Games'])

    for tm in sorted(team_data):
        for season in sorted(team_data[tm]):
            coaches = team_data[tm][season]
            names = [entry['coach'] for entry in coaches]
            summary_writer.writerow([tm, season, ', '.join(names)])
            if coaches:
                first_writer.writerow([tm, season, coaches[0]['coach']])
                max_entry = max(coaches, key=lambda entry: entry['games'])
                max_games_writer.writerow([tm, season, max_entry['coach'], max_entry['games']])
            else:
                first_writer.writerow([tm, season, ''])
                max_games_writer.writerow([tm, season, '', 0])

for tm in sorted(team_data):
    print(tm)
    for season in sorted(team_data[tm]):
        names = ', '.join(entry['coach'] for entry in team_data[tm][season])
        print(f"  {season}: {names}")
    print()
