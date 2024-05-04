import argparse
import os
from helpers import generate_data


def main(args):
    print("Generating validation data ... ...")
    generate_data(
        N=args.N, 
        num_samples=args.num_samples_val, 
        max_targets=args.max_targets,
        folder_path=os.path.join(args.output_dir, 'data/val')
    )

    print("Generating training data ... ...")
    generate_data(
        N=args.N, 
        num_samples=args.num_samples_train, 
        max_targets=args.max_targets,
        folder_path=os.path.join(args.output_dir, 'data/train')
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset for antenna signal processing.')
    parser.add_argument('--output_dir', type=str, default='../', 
                        help='Base directory for output data')
    parser.add_argument('--num_samples_val', type=int, default=1024, 
                        help='Number of validation samples to generate')
    parser.add_argument('--num_samples_train', type=int, default=100000, 
                        help='Number of training samples to generate')
    parser.add_argument('--N', type=int, default=10, 
                        help='Number of antenna elements')
    parser.add_argument('--max_targets', type=int, default=3, 
                        help='Maximum number of targets per sample')
    args = parser.parse_args()
    main(args)