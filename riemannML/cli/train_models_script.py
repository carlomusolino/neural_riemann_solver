import argparse
from .train_models_main import train_all_models_main

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the neural Riemann solver model components."
    )

    parser.add_argument(
        "--seed", type=int, required=True,
        help="Random seed for dataset generation, weight initialization, and batching. Ensures reproducibility."
    )

    parser.add_argument(
        "--n_neurons", type=int, required=True,
        help="Number of neurons in each hidden layer of the pressure predictor MLPs."
    )

    parser.add_argument(
        "--n_dsets", type=int, required=True,
        help="Number of training samples per component (e.g., shock-shock, rarefaction-shock, etc.)."
    )

    parser.add_argument(
        "--batch", type=int, required=True,
        help="Batch size for training (number of samples per gradient update)."
    )

    parser.add_argument(
        "--basedir", type=str, default="../models",
        help="Base directory where trained models and logs will be saved (default: '../models')."
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-2,
        help="Initial learning rate for the optimizer (default: 1e-2)."
    )

    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of hidden layers in each MLP (default: 2)."
    )

    args = parser.parse_args()


    train_all_models_main(args.seed, args.n_dsets, args.batch, args.n_layers, args.n_neurons, args.learning_rate, args.basedir)


if __name__ == "__main__":
    main()
