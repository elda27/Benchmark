import argparse
import dataset
import model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, choices=dataset.get_dataset_names()
    )
    parser.add_argument(
        '--model', type=str, choices=model.get_trainer_list(),
    )
    parser.add_argument()
    args = parser.parse_args()

    trainer_registry = model.get_register_tariner(args.model)


if __name__ == '__main__':
    main()
