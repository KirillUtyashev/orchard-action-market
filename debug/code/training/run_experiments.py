import argparse
import sys

from debug.code.core.config import load_config
from debug.code.training.helpers import set_all_seeds
from debug.code.training.learning import Learning


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Dot-notation overrides, e.g. train.alpha=0.001 network.CNN=true")
    return parser.parse_args(args)


def run_one(config_path: str, overrides: list[str]):
    cfg = load_config(config_path, overrides=overrides)

    set_all_seeds(seed=cfg.train.seed)
    algo = Learning(cfg)
    history = algo.train()

    return {"lr": str(cfg.train.alpha), "seed": cfg.train.seed, "history": history}


def main(args):
    args = parse_args(args)
    run_one(args.config, args.override)


if __name__ == "__main__":
    main(sys.argv[1:])
