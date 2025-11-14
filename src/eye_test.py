import sys
from pathlib import Path
from typing import List

import pandas as pd

# Hard-coded list of player identifiers to inspect.
PLAYERS_OF_INTEREST: List[str] = [
    'jamesle01',
    'curryst01',
    'duranke01',
]


def _prompt_for_csv_path() -> Path:
    """Prompt the user for a CSV path until a valid file is provided."""
    while True:
        try:
            response = input("Enter path to the player CSV file: ").strip()
        except EOFError:
            raise SystemExit("No input received. Exiting.")

        if not response:
            print("Path cannot be empty. Please try again.", file=sys.stderr)
            continue

        path = Path(response).expanduser().resolve()
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue
        if path.suffix.lower() != '.csv':
            print(f"Expected a CSV file. Received: {path}", file=sys.stderr)
            continue

        return path


def filter_players(csv_path: Path) -> Path:
    """Filter the provided CSV to only the players of interest."""
    df = pd.read_csv(csv_path)
    column = 'Player-additional'
    if column not in df.columns:
        raise KeyError(f"Required column '{column}' not found in {csv_path}")

    filtered = df[df[column].isin(PLAYERS_OF_INTEREST)].copy()
    output_path = csv_path.with_name(f"{csv_path.stem}_eye_test.csv")
    filtered.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    csv_path = _prompt_for_csv_path()
    output_path = filter_players(csv_path)
    print(f"Filtered data saved to: {output_path}")


if __name__ == "__main__":
    main()
