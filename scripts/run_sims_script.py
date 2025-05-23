import argparse 

from run_sims_main import main 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_neurons", type=int, required=True)
    parser.add_argument("--n_dsets", type=int, required=True)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--basedir", type=str, default='../models')
    args = parser.parse_args()

    main(args.seed, args.n_dsets, args.n_layers, args.n_neurons, args.basedir )