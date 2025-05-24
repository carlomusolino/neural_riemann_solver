import argparse 
from .run_sims_ensemble_main import run_sims_ensemble

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neurons", type=int, required=True)
    parser.add_argument("--n_dsets", type=int, required=True)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--seed_combination", type=int, nargs='+', required=True)
    parser.add_argument("--basedir", type=str, default="../model")
    
    args = parser.parse_args()

    run_sims_ensemble(args.n_dsets, args.n_layers, args.n_neurons, args.seed_combination, args.basedir)

if __name__ == "__main__":
    main()
