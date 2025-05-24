import argparse
from .train_models_main import train_all_models_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_neurons", type=int, required=True)
    parser.add_argument("--n_dsets", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--basedir", type=str, default="../models")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--n_layers", type=int, default=2)
    args = parser.parse_args()

    train_all_models_main(args.seed, args.n_dsets, args.batch, args.n_layers, args.n_neurons, args.learning_rate, args.basedir)


if __name__ == "__main__":
    main()
