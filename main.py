# keeping the command line entry point for the full analysis here
import argparse
# importing the experiment stages
from src.stability_analysis import run as run_stability
from src.reflectance_baseline import run as run_reflectance
from src.visualization import run as run_figures


# parsing command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Illumination invariant superpixels: stability, reflectance baseline and illumination grids.")
    # selecting only the stability analysis stage
    parser.add_argument("--stability", action="store_true",
                        help="run only the stability analysis and write the metrics CSV")
    # selecting only the reflectance baseline stage
    parser.add_argument("--reflectance", action="store_true",
                        help="run only the reflectance baseline comparison")
    # selecting only the figure generation stage
    parser.add_argument("--figures", action="store_true",
                        help="run only the illumination grid figure generation")
    return parser.parse_args()


# running the requested stages
def main():
    args = parse_args()
    # running all stages when no stage flag is given
    run_all = not (args.stability or args.reflectance or args.figures)
    # running the stability analysis stage
    if run_all or args.stability:
        run_stability()
    # running the reflectance baseline stage
    if run_all or args.reflectance:
        run_reflectance()
    # running the figure generation stage
    if run_all or args.figures:
        run_figures()


if __name__ == "__main__":
    main()
