import argparse
import dataset
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    with open(args.config) as fp:
        config = json.load(fp)

    config = create_config(**config)


def create_config(**kwargs):
    kwargs['']


if __name__ == '__main__':
    main()
