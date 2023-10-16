import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

import utils
import prompt as pt
from sklearn.cluster import SpectralClustering, KMeans


def gen_proto(args, feature_extractor, dataset, prompts=None, load_path='', name=''):
    """
    Generate class-wise prototype using class mean vector.
    """
    
    if load_path:
        prototype_dict = torch.load(load_path)
        prototypes = prototype_dict['mean'].cuda()
        prototypes_var = prototype_dict['var'].cuda()
        prototypes_similarity = prototype_dict['sim']
        all_features = prototype_dict['fea']
    else:
        with torch.no_grad():
            if prompts is not None:
                feature_extractor = pt.ProTLearner(args, feature_extractor)
            feature_extractor.cuda().eval()
            prototypes, prototypes_var, prototypes_similarity, all_features = [], [], [], []
            tqdm_gen = tqdm.tqdm(range(args.total_nc))
            tqdm_gen.set_description('Generate Prototypes')
            for class_id in tqdm_gen:
                tqdm_gen.set_postfix({'prototype id':class_id})
                # insert right prompter
                if prompts is not None:
                    feature_extractor.prompts = nn.Parameter(prompts[class_id, :])
                    feature_extractor = feature_extractor.eval()

                features, var, mean, cosine_similarity = \
                gen_single_proto(args, feature_extractor, class_id, dataset)

                all_features.append(features)
                prototypes.append(mean)
                prototypes_var.append(var)
                prototypes_similarity.append(cosine_similarity.squeeze())

            prototypes = torch.stack(prototypes, dim=0)  # shape(num_class, feature_dim)
            prototypes_var = torch.stack(prototypes_var, dim=0)  # shape(num_class, feature_dim)
            prototypes_similarity = prototypes_similarity  # list: len(num_classes) 
            all_features = all_features  # list: len(num_classes) 

            # save prototypes and their relatives
            args.save_helper.save_prototype({'mean':prototypes, 'var':prototypes_var, \
                                            'sim':prototypes_similarity, 'fea':all_features}, name=name)

    print('Prototypes size: {}'.format(prototypes.size()))
    return prototypes, prototypes_var, prototypes_similarity, all_features


def gen_single_proto(args, feature_extractor, class_id, dataset):
    with torch.no_grad():
        feature_list = []
        dataset.set_classes([class_id])
        data_loader = DataLoader(dataset=dataset,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 batch_size=args.bs,
                                 drop_last=False)
        for _, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            outputs = feature_extractor(imgs)
            feature_list.append(outputs)
        features = torch.cat(feature_list, dim=0)
        var, mean = torch.var_mean(features, dim=0)
        cosine_similarity = utils.cosine_similarity(mean.unsqueeze(0), features)
    return features, var, mean, cosine_similarity


def gen_mc_proto(args, all_features, load_path='', name=''):
    if load_path:
        multi_c_dict = torch.load(load_path)
        multi_c_proto, multi_c_proto_var, index_mapping = \
        multi_c_dict['prototypes'], multi_c_dict['prototypes_var'], multi_c_dict['mapping']
    else:
        multi_c_proto, multi_c_proto_var = [], []
        for _, cur_class in enumerate(args.class_seq):
            input_data = all_features[cur_class]
            cur_proto, cur_proto_var, _, _ = \
            gen_single_mc_proto(args, input_data)
            multi_c_proto.append(cur_proto)
            multi_c_proto_var.append(cur_proto_var)
        # adjust order
        multi_c_proto = utils.adjust_order(multi_c_proto, args.class_seq)
        multi_c_proto_var = utils.adjust_order(multi_c_proto_var, args.class_seq)
        # prepare multi_c_proto and corresponding gt tensor
        index_mapping = get_index_mapping(multi_c_proto)
        # concate 
        multi_c_proto = torch.cat(multi_c_proto, dim=0).cuda() 
        multi_c_proto_var = torch.cat(multi_c_proto_var, dim=0).cuda() 
        index_mapping = torch.cat(index_mapping, dim=0).squeeze().cuda()  # containing class labels
        try:
            args.save_helper.save_prototype({'prototypes':multi_c_proto, 'prototypes_var':multi_c_proto_var, \
                                            'mapping':index_mapping}, name=name)
        except:
            pass
    print('Multi-centroid prototypes size: {}'.format(multi_c_proto.size()))
    return multi_c_proto, multi_c_proto_var, index_mapping


def get_index_mapping(multi_c_proto):
    """
    Generate the mapping relation for multi-centriods prototypes,
    the order of the input need to be sorted ascendingly according to the labels.
    """
    assert type(multi_c_proto) == list, print('Wrong data structure')
    index_mapping = []
    for class_idx in range(len(multi_c_proto)):
        cur_protos = multi_c_proto[class_idx].squeeze()
        if len(cur_protos.size()) == 1:
            cur_mapping = torch.full((1,1), class_idx)
        else:
            cur_mapping = torch.full((len(cur_protos),1), class_idx)
        index_mapping.append(cur_mapping)
    return index_mapping


def gen_single_mc_proto(args, input_data):
    cur_proto, cur_proto_var, cur_proto_sim, cur_features = [], [], [], []
    if args.gen_proto_mode == 'spectral':
        affinity_matrix = utils.cosine_similarity(input_data, input_data)
        clustering = SpectralClustering(n_clusters=args.mc_num, \
        assign_labels='discretize', affinity='precomputed', n_init=10, \
        random_state=args.seed)
        affinity_matrix = affinity_matrix.cpu().numpy()
        clustering.fit_predict(affinity_matrix)
    elif args.gen_proto_mode == 'kmeans':
        clustering = KMeans(n_clusters=args.mc_num, random_state=args.seed)
        clustering.fit(input_data.cpu().numpy())
    else:
        raise NotImplementedError

    for label in range(args.mc_num):
        feature = input_data[clustering.labels_ == label, :]
        if not torch.is_tensor(feature):
            feature = torch.tensor(feature).cuda()
        var, mean = torch.var_mean(feature, dim=0)
        sim = utils.cosine_similarity(mean.unsqueeze(0), feature)
        cur_proto.append(mean)
        cur_proto_var.append(var)
        cur_proto_sim.append(sim)
        cur_features.append(feature)
    cur_proto = torch.stack(cur_proto, dim=0)
    cur_proto_var = torch.stack(cur_proto_var, dim=0)
    return cur_proto, cur_proto_var, cur_proto_sim, cur_features