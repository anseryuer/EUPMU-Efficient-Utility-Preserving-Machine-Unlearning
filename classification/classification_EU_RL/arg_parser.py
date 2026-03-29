import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    # parser.add_argument(
    #     "--data_dir",
    #     type=str,
    #     default="./tiny-imagenet-200",
    #     help="dir to tiny-imagenet",
    # )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default=None,
        help="Path to cached training labels tensor for ImageNet loaders",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default=None,
        help="Path to cached validation labels tensor for ImageNet loaders",
    )
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=str, default="0", help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )

    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", type=int, default=-1, help="Specific class to forget"
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")

    parser.add_argument("--path", default=None, type=str, help="mask matrix")
    parser.add_argument('--num_iter', default=None, type=int, help='the number of iteration')
    parser.add_argument(
        "--mask_ratio", type=float, default=0.5, help="mask ratio for unlearning"
    )
    ##################################### Attack setting #################################################
    parser.add_argument(
        "--attack", type=str, default="backdoor", help="method to unlearn"
    )
    parser.add_argument(
        "--trigger_size",
        type=int,
        default=4,
        help="The size of trigger of backdoor attack",
    )


    ######################################################
    parser.add_argument('--times_para', type=int, default=1,
                    help='times of selected parameter')
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    parser.add_argument("--tuning_head", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--select_head", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--only_trainForgetSet", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--only_trainForgetSet_and_samesizeOfretain", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--retainwithAllParamUpdate", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--ovrelap", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--reverse", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--mtl", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--noGPS", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--SelecteParameterRandomLabelForgetSet", action = "store_true",default = False, help="whether tune head")
    parser.add_argument("--mtl_method", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--eu_w_lr", default=3, type=float, help="learning rate for weight EUPMU, the bigger the faster weight changes to retain loss change")
    parser.add_argument("--eu_error", default=0.03, type=float, help="the error of weight update in EUPMU, more positive means more focus on retain")
    parser.add_argument("--weight_init", default=0.0, type=float, help="initial weight for EU or RL")
    parser.add_argument("--cheby_retain_weight", default=1.0, type=float, help="Chebyshev weight for the retain loss")
    parser.add_argument("--cheby_forget_weight", default=1.0, type=float, help="Chebyshev weight for the forget loss")
    parser.add_argument("--cheby_retain_ref", default=0.0, type=float, help="Chebyshev reference value for the retain loss")
    parser.add_argument("--cheby_forget_ref", default=0.0, type=float, help="Chebyshev reference value for the forget loss")
    parser.add_argument("--cheby_rho", default=1e-3, type=float, help="Augmentation coefficient for Chebyshev scalarization")
    parser.add_argument("--omd_tch_retain_weight", default=1.0, type=float, help="OMD-TCH weight for the retain loss")
    parser.add_argument("--omd_tch_forget_weight", default=1.0, type=float, help="OMD-TCH weight for the forget loss")
    parser.add_argument("--omd_tch_retain_ref", default=0.0, type=float, help="OMD-TCH reference value for the retain loss")
    parser.add_argument("--omd_tch_forget_ref", default=0.0, type=float, help="OMD-TCH reference value for the forget loss")
    parser.add_argument("--omd_tch_eta", default=0.1, type=float, help="Mirror descent step size for OMD-TCH")
    parser.add_argument("--omd_tch_rho", default=0.0, type=float, help="Optional augmentation coefficient for OMD-TCH; paper-consistent default is 0.0")
    parser.add_argument("--percent_pruning_min", default=0, type=float)
    parser.add_argument("--percent_pruning_max", default=100, type=float)

    return parser.parse_args()
