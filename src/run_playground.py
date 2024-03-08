import argparse

import torch

from utils.interactive_visualizations.playground import Playground


def parse_args():
    parser = argparse.ArgumentParser(description='run playground')
    # parser.add_argument('model_path', type=str, default=MODEL_PATH, nargs='?', help='The model to load')

    return parser.parse_args()


def main():
    # args = parse_args()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    window = Playground()
    window.run()


if __name__ == '__main__':
    main()
