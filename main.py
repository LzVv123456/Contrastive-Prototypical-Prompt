import sys 
import copy
import argparse
import time
import numpy as np
import timm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models

import utils
import datasets 
import transformers
import prototype as prot
import train
import test


def get_args_parser():
    parser = argparse.ArgumentParser(description='Prototype Prompt for Continual Learning')

    # general -------------------------------------------------------------------------------------
    parser.add_argument('--epochs', default=50, type=int, help='total number of epochs to run')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='warmup number of epochs')
    parser.add_argument('--bs', default=256, type=int, help='batch-size for training')
    parser.add_argument('--numerical_order', default=True, type=utils.bool_flag, help='whether fix the class order')
    parser.add_argument('--finetune_vit', default=False, type=utils.bool_flag, help='whether allow fintune vit')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet_r', '5datasets'], help='Dataset to use')
    parser.add_argument('--loss', default='protocon', type=str, help='loss function')
    parser.add_argument('--total_nc', default=100, type=int, help='total class number for the dataset')
    parser.add_argument('--fg_nc', default=10, type=int, help='the number of classes in the first task')
    parser.add_argument('--task_num', default=9, type=int, help='the number of tasks')

    # optm ---------------------------------------------------------------------------------------
    parser.add_argument('--p_lr', default=0.001, type=float, help='prompt initial learning rate')
    parser.add_argument('--p_min_lr', default=1e-6, type=float, help='prompt minumum learning rate')
    parser.add_argument('--p_optm', default='adamw', type=str, choices=['sgd', 'adamw', 'adam'], help='prompt optimizer')
    parser.add_argument('--p_wd', default=5e-4, type=float, help='prompt weight decay')
    parser.add_argument('--p_wd_end', default=1e-2, type=float, help='prompt end weight decay')
    parser.add_argument('--p_schd', default='cosine', type=str, help='prompt scheduler')
    parser.add_argument('--h_lr', default=0.001, type=float, help='head initial learning rate')
    parser.add_argument('--h_min_lr', default=1e-6, type=float, help='head minumum learning rate')
    parser.add_argument('--h_optm', default='adamw', type=str, choices=['sgd', 'adamw', 'adam'], help='head optimizer')
    parser.add_argument('--h_wd', default=5e-4, type=float, help='head weight decay')
    parser.add_argument('--h_wd_end', default=1e-2, type=float, help='head end weight decay')
    parser.add_argument('--h_schd', default='cosine', type=str, help='head scheduler')

    # cpp -----------------------------------------------------------------------------------
    parser.add_argument('--add_pos_proto', default=False, type=utils.bool_flag, help='whether add positive prototypes')
    parser.add_argument('--add_neg_proto', default=True, type=utils.bool_flag, help='whether add negative prototypes')
    parser.add_argument('--add_uniformity', default=False, type=utils.bool_flag, help='whether add uniformity')
    parser.add_argument('--temp', default=0.6, type=float, help='temperature')
    parser.add_argument('--neg_temp', default=0.6, type=float, help='temperature for neg components')
    parser.add_argument('--proto_aug_bs', default=256, type=int, help='bs for proto augmentation')
    parser.add_argument('--proto_trans', default=True, type=utils.bool_flag, help='whether use proto transformaton')

    # Prototype --------------------------------------------------------------------------------------
    parser.add_argument('--use_mc_proto', default=True, type=utils.bool_flag, help='whether use multi-centroid prototype')
    parser.add_argument('--mc_num', default=5, type=int, help='number of centroids for each prototype')
    parser.add_argument('--similarity', default='cosine', type=str, choices=['l1','l2','cosine'], help='way to compute similarity(distance)')
    parser.add_argument('--gen_proto_mode', default='spectral', type=str, choices=['spectral', 'kmeans'], help='clustering method for generating multi-centroid prototypes')
   
    parser.add_argument('--k_p_path', default='', type=str, help='path for key prototype')
    parser.add_argument('--k_mc_p_path', default='', type=str, help='path for multi-centroid key prototype')
    parser.add_argument('--v_p_path', default='', type=str, help='path for value prototype')
    parser.add_argument('--v_mc_p_path', default='', type=str, help='path for multi-centroid value prototype')

    # prompt ----------------------------------------------------------------------------------------
    parser.add_argument('--init_with_prev_prompt', default=True, type=utils.bool_flag, help='whether init prompt from previous session')
    parser.add_argument('--prompt_path', default='', type=str, help='path for saved prompts')
    parser.add_argument('--l_p', default=1, type=int, help='prompt length')
    parser.add_argument('--d_p', default=True, type=utils.bool_flag, help='whether use deep prompt')
    parser.add_argument('--topk', default=3, type=int, help='topk retrieve for prompt')

    # arch head ------------------------------------------------------------------------------------------
    parser.add_argument('--add_mlp', default=True, type=utils.bool_flag, help='whether add mlp neck')
    parser.add_argument('--add_cls', default=False, type=utils.bool_flag, help='whether add linear classifier')
    parser.add_argument('--incre_cls', default=False, type=utils.bool_flag, help='whether use incremental classifier')
    parser.add_argument('--cls_dim', default=1, type=int, help='output dimension for linear classifier')
    parser.add_argument('--mlp_layer_num', default=3, type=int, help='layers of mlp neck')
    parser.add_argument('--mlp_bottleneck_dim', default=768, type=int, help='output dimesion of mlp neck')
    parser.add_argument('--mlp_hidden_dim', default=2048, type=int, help='hidden dimesion of mlp neck')
    parser.add_argument('--mlp_use_bn', default=False, type=utils.bool_flag, help='Whether or not to weight normalize the linear classifier')
    parser.add_argument('--normalize', default=True, type=utils.bool_flag, help='whether normalize vectors')

    # Backbone ---------------------------------------------------------------------------------------
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model')
    parser.add_argument('--pretrain_method', default='1k', type=str, choices=['dino', 'mae', 'deit', '1k', '21k',], help='load weights trained with different methods')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    # Misc ---------------------------------------------------------------------------------------
    parser.add_argument('--data_path', default='./data', type=str, help='Please specify path to the folder where data is saved.')
    parser.add_argument('--output_dir', default='./exps_results', type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--infer_path', default='', type=str, help='path for inferring specific exp')
    parser.add_argument('--exp_name', default='', type=str, help='experiment name.')
    parser.add_argument('--seed', default=77, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')

    return parser


def set_tasks(args):
    # set task size and class order
    if args.task_num > 0:
        # number of classes in each incremental step
        args.task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    else:
        args.task_size = 0
    if args.dataset == '5datasets': 
        # always set fixed numerical order for 5-datasets for convienince
        args.class_seq = np.array(range(args.total_nc))
    else:
        if args.numerical_order:
            args.class_seq = np.array(range(args.total_nc))
        else:
            # random permutation
            args.class_seq = np.random.permutation(range(args.total_nc))

    # generate class-2-session map
    class_session_map = {}
    for idx, class_id in enumerate(args.class_seq):
        if idx < args.fg_nc:
            class_session_map[class_id] = 0
        else:
            class_session_map[class_id] = ((idx-args.fg_nc)//args.task_size)+1
    args.class_session_map = class_session_map


def build_model(args):
    if args.pretrain_method == 'dino':
        if "vit" in args.arch:
            model = transformers.dino_vit.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
            print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
        elif "xcit" in args.arch:
            model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch](num_classes=0)
            model.fc = nn.Identity()
        else:
            print(f"Architecture {args.arch} non supported")
            sys.exit(1)
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    elif args.pretrain_method == 'mae':
        model = transformers.mae_vit.vit_base_patch16(global_pool=True)
        checkpoint = torch.load('./transformers/mae_finetuned_vit_base.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        transformers.mae_vit.interpolate_pos_embed(model, checkpoint_model)
        _ = model.load_state_dict(checkpoint_model, strict=False)
        model.head = nn.Identity()

    elif args.pretrain_method == 'deit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        model.head = nn.Identity()

    elif args.pretrain_method == '1k':
        model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
        model.head = nn.Identity()
    
    elif args.pretrain_method == '21k':
        model = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True)
        model.head = nn.Identity()

    else:
        raise NotImplementedError

    return model


def main(args):
    # set up misc
    utils.fix_random_seeds(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set task size and class order
    set_tasks(args)

    # init save_helper
    args.save_helper = utils.save_helper(args)

    # build network
    feature_extractor = build_model(args)

    # prepare dataset
    train_dataset, gen_proto_dataset, test_dataset = datasets.prepare_dataset(args)

    # learn task-specific prompts
    if args.prompt_path:
        prompts = torch.load(args.prompt_path).cuda()
    else:
        trainer = train.Trainer(args, copy.deepcopy(feature_extractor), train_dataset, gen_proto_dataset)

        prompts = trainer.train_discriminate()
        # save prototype prompts
        args.save_helper.save_prompt(prompts)
        # saved finetuned vit
        if args.finetune_vit:
            args.save_helper.save_torch(trainer.vit_model, 'finetune_vit')

    # build model
    if args.finetune_vit:
        feature_extractor = torch.load(args.output_dir + '/finetune_vit.pt')
    else:
        feature_extractor = build_model(args)

    # generate key prototypes
    k_protos, k_protos_var, k_protos_sim, k_features = \
    prot.gen_proto(args, feature_extractor, \
    gen_proto_dataset, load_path=args.k_p_path, name='k')
    # generate key multi-centroid prototypes
    k_mc_protos, k_mc_protos_var, k_mc_mapping = \
    prot.gen_mc_proto(args, k_features, \
    load_path=args.k_mc_p_path, name='k_mc')

    # generate value prototypes
    v_protos, v_protos_var, v_protos_sim, v_features = \
    prot.gen_proto(args, copy.deepcopy(feature_extractor), gen_proto_dataset, \
                   prompts=prompts, load_path=args.v_p_path, name='v') 
    # generate value multi-centroid prototypes
    v_mc_protos, v_mc_protos_var, v_mc_mapping = \
    prot.gen_mc_proto(args, v_features, \
    load_path=args.v_mc_p_path, name='v_mc')

    # test 
    test.test(args, feature_extractor, test_dataset, prompts, \
              k_protos, k_mc_protos, k_mc_mapping, \
              v_protos, v_mc_protos, v_mc_mapping)


if __name__ == "__main__":
    s_t = time.time()
    parser = get_args_parser()
    args = parser.parse_args()
    torch.set_printoptions(precision=4)
    main(args)
    e_t = time.time()
    print('Time Usage:{}'.format(np.around((e_t-s_t)/3600, 2)))
