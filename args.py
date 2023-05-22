import argparse

def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ########################################
    ########################################
    # Train arguments
    ########################################
    ########################################

    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs for training"
    )

    parser.add_argument(
        "--gpu", type=bool, default=False, help="whether to use gpu acceleration"
    )

    parser.add_argument(
        "--train-mode", type=str, default="train", help="test or train"
    )

    ########################################
    ########################################
    # Loss arguments
    ########################################
    ########################################

    parser.add_argument(
        "--loss-name", type=str, default="CrossEntropyLoss", help="loss function of the train"
    )

    ########################################
    ########################################
    # Dataset arguments
    ########################################
    ########################################

    parser.add_argument(
        "--dataset-root", type=str, default="./dataset/Places2_simp", help="root path to data directory"
    )

    parser.add_argument(
        "--width", type=int, default=224, help="width of input image"
    )

    parser.add_argument(
        "--height", type=int, default=224, help="height of input image"
    )

    parser.add_argument(
        "--batch-size", type=int, default=256, help="batch size for training"
    )

    parser.add_argument(
        "--num-workers", type=int, default=4, help="number of workers for data loading"
    )

    parser.add_argument(
        "--train-size-len", type=float, default=0.8, help="Size of the training set"
    )

    parser.add_argument(
        "--shuffle", type=bool, default=True, help="shuffle the data"
    )

    ########################################
    ########################################
    # Optimizer arguments
    ########################################
    ########################################
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate for training"
    )

    parser.add_argument(
        "--last-lr", type=float, default=0.001, help="learning rate for training"
    )

    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum for optimizer"
    )

    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="weight decay for optimizer"
    )

    parser.add_argument(
        "--dampening", type=float, default=0.0, help="dampening for optimizer"
    )

    parser.add_argument(
        "--nesterov", type=bool, default=False, help="nesterov for optimizer"
    )

    parser.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 for optimizer"
    )

    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for optimizer"
    )

    parser.add_argument(
        "--eps", type=float, default=1e-08, help="epsilon for optimizer"
    )

    parser.add_argument(
        "--amsgrad", type=bool, default=False, help="amsgrad for optimizer"
    )

    parser.add_argument(
        "--alpha", type=float, default=0.99, help="alpha for optimizer"
    )

    parser.add_argument(
        "--centered", type=bool, default=False, help="centered for optimizer"
    )

    parser.add_argument(
        "--lr-decay", type=float, default=0.0, help="lr decay for optimizer"
    )

    parser.add_argument(
        "--initial-accumulator-value", type=float, default=0.0, help="initial accumulator value for optimizer"
    )
    
    ########################################
    ########################################
    # Model arguments
    ########################################
    ########################################

    parser.add_argument(
        "--model", type=str, default="resnet34", help="model for training"
    )

    parser.add_argument(
        "--num-classes", type=int, default=40, help="number of classes"
    )

    parser.add_argument(
        "--num-samples", type=int, default=5, help="number of samples"
    )

    ########################################
    ########################################
    # Log arguments
    ########################################
    ########################################
    parser.add_argument(
        "--save-path", type=str, default="logs/demo03", help="path to save results"
    )

    parser.add_argument(
        "--result-name", type=str, default="trained-model", help="model and log name"
    )

    return parser.parse_args()

# Make optimizer arguments
def make_opti_args():
    args = argument_parser()
    opti_args = {
        "lr": args.lr,
        "last_lr": args.last_lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "dampening": args.dampening,
        "nesterov": args.nesterov,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "amsgrad": args.amsgrad,
        "alpha": args.alpha,
        "centered": args.centered,
        "lr_decay": args.lr_decay,
        "initial_accumulator_value": args.initial_accumulator_value
    }
    return opti_args
