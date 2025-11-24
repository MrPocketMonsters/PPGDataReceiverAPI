import pandas
from datetime import datetime as Datetime
import os
from pathlib import Path
import re

TIMESTAMP_KEY = 'TIMESTAMP'
RED_KEY = 'RED'
IR_KEY = 'IR'
GREEN_KEY = 'GREEN'
DELTA_END = '_DELTA'

def ppg_dict_to_dataframe(ppg_dict: dict) -> pandas.DataFrame:
    """Converts a PPG data dictionary to a pandas DataFrame."""

    def deltas_to_values(deltas: list[int]) -> list[int]:
        """Converts a list in delta format to absolute format."""
        if not deltas:
            return deltas

        values = [deltas[0]]
        for delta in deltas[1:]:
            values.append(values[-1] + delta)

        return values

    # Determine if the data is in delta format or invalid.
    is_delta_format = False
    if ppg_dict.get(TIMESTAMP_KEY) is None:
        if ppg_dict.get(TIMESTAMP_KEY + DELTA_END):
            is_delta_format = True
        else:
            raise ValueError("No timestamp data found in dictionary.")

    # Prepare keys based on format.
    timestamp_key = TIMESTAMP_KEY + (DELTA_END if is_delta_format else '')
    red_key = RED_KEY + (DELTA_END if is_delta_format else '')
    ir_key = IR_KEY + (DELTA_END if is_delta_format else '')
    green_key = GREEN_KEY + (DELTA_END if is_delta_format else '')

    # Extract data from the dictionary.
    new_dict = {
                key.split('_')[0]: ppg_dict.get(key, [])
                        for key in [timestamp_key, red_key, ir_key, green_key]
            }

    # Convert from delta format to absolute values if necessary.
    if is_delta_format:
        for key in new_dict:
            deltas = new_dict[key]
            new_dict[key] = deltas_to_values(deltas)

    index = new_dict.pop(TIMESTAMP_KEY)
    return pandas.DataFrame(new_dict, index=index)

def store_ppg_dataframe_to_csv(folder: str, df: pandas.DataFrame) -> str:
    """Stores the PPG DataFrame to a CSV file with a timestamped filename and returns the file path."""
    timestamp_prefix = Datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = __store_ppg_dataframe_to_csv_with_prefix__(folder, timestamp_prefix, df)
    return filepath

def load_top_n_csv_to_dataframe(folder: str, top_n: int) -> pandas.DataFrame | None:
    """Loads the top N most recent PPG CSV files from the specified folder and combines them into a single DataFrame."""
    folder = Path(folder).resolve()

    # list all files with pattern '<timestamp>_ppg.csv'
    files: list[Path] = [p for p in folder.iterdir() if p.is_file() and p.name.endswith("_ppg.csv")]
    pairs: list[tuple[Datetime, Path]] = []  # (datetime, path)
    for path in files:
        datetime = __parse_timestamp_from_name__(path.name)
        if datetime is not None:
            pairs.append((datetime, path))

    if not pairs:
        print(f"Warning: did not find any files with pattern '<timestamp>_ppg.csv' in {folder}")
        return None

    # order by timestamp descending
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Take top_n most recent and reverse to chronological order
    selected = [path for (_, path) in reversed(pairs[:top_n])]
    print(f"Selected {len(selected)} files (most recent).")

    # Parse and combine dataframes
    dfs: list[pandas.DataFrame] = []
    for path in selected:
        try:
            dfs.append(pandas.read_csv(path, header=0, index_col=0))
        except Exception as e:
            print(f"Error: could not read {path.name}: {e}")

    if not dfs:
        print("Warning: No data could be read from the selected files.")
        return None

    return pandas.concat(dfs, axis=0)

def __store_ppg_dataframe_to_csv_with_prefix__(folder: str, prefix: str, df: pandas.DataFrame) -> str:
    """Stores the PPG DataFrame to a CSV file and returns the file path."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{prefix}_ppg.csv"
    filepath = os.path.join(folder, filename)

    df.to_csv(filepath)
    return filepath

def __parse_timestamp_from_name__(name: str) -> Datetime | None:
    """Parses a timestamp from a filename with format '<timestamp>_ppg.csv'."""
    groups = re.match(r"^(?P<ts>[^_]+)_ppg\.csv$", name)
    if not groups:
        return None
    timestamp = groups.group("ts")

    try:
        return Datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%SZ")
    except ValueError:
        return None
