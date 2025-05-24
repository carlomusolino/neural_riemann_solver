import argparse 

from .run_sims_main import run_sims 

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run shocktube tests using different Riemann solvers and resolutions."
    )

    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed used for model selection and reproducibility.")

    parser.add_argument("--n_neurons", type=int, required=True,
                        help="Number of neurons per hidden layer in the pressure predictor network.")

    parser.add_argument("--n_dsets", type=int, required=True,
                        help="Number of training samples used for model training.")

    # First-order switch (with both positive and negative form)
    parser.add_argument("--do_first_order", dest="do_first_order", action="store_true",
                        help="Run first-order simulations (default: enabled).")
    parser.add_argument("--no_first_order", dest="do_first_order", action="store_false",
                        help="Disable first-order simulations.")
    parser.set_defaults(do_first_order=True)

    # Second-order switch
    parser.add_argument("--do_second_order", action="store_true",
                        help="Run second-order simulations using WENO and midpoint integration.")

    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[50, 100, 200, 400, 800, 1600, 3200],
                        help="List of grid resolutions to simulate.")

    parser.add_argument("--solvers", type=str, nargs="+",
                        default=['hlle', 'hllc', 'ml'],
                        help="List of Riemann solvers to test. Options: 'hlle', 'hllc', 'ml'.")

    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of hidden layers in the pressure predictor network.")

    parser.add_argument("--basedir", type=str, default='../models',
                        help="Base directory containing the trained model(s).")

    args = parser.parse_args()


    run_sims(
        args.seed, 
        args.n_dsets, 
        args.n_layers, 
        args.n_neurons, 
        args.do_first_order, 
        args.do_second_order, 
        args.resolutions,
        args.solvers,
        args.basedir )

if __name__=="__main__":
    main()
