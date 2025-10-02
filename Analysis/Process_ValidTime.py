import logging
from argparse import ArgumentParser
from pathlib import Path

from drrc.analysis import AutomaticPostprocessing
from drrc.config import Config


def main():
    parser = ArgumentParser(
        prog="Process_ValidTime",
        description="Processing of ValidTime output",
    )
    parser.add_argument("path", help="Path to data (can contain wildcards)")
    args = parser.parse_args()
    path_list = (Config.get_git_root() / Path("Data")).glob(args.path)

    logging.basicConfig(level=logging.INFO)
    # Concatenate and generate statistics for all valid times
    for path in path_list:
        AutomaticPostprocessing(path).auto_concatenate()
        AutomaticPostprocessing(path).auto_statisticsgeneration()


if __name__ == "__main__":
    main()
