import argparse
import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=dataset.get_dataset_names(), type=str
    )
    parser.add_argument(
        '--model', type=str
    )


if __name__ == '__main__':
    main()
