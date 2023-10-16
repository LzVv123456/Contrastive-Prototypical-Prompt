import tqdm
import pprint
import numpy as np
import prompt as pt
import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(args, feature_extractor, test_dataset, prompts, \
         k_protos, k_mc_protos, k_mc_mapping, \
         v_protos, v_mc_protos, v_mc_mapping):

    with torch.no_grad():
        # init model
        feature_extractor=feature_extractor.cuda().eval()
        prompter = pt.ProTLearner(args, feature_extractor).cuda().eval()

        # init helper
        cl_helper = utils.continual_metric_helper(args)
        cl_helper_prompt = utils.continual_metric_helper(args)

        tqdm_gen_session = tqdm.tqdm(range(args.task_num+1)) 
        tqdm_gen_session.set_description('Session Loop')

        # count num of retrieved prompts
        count_prompt_collection = [] 
        count_img_collection = []

        # count accident retrieve
        total_accident_retrieve = 0
        total_concurrent_retrieve = 0
        total_img_count = 0

        for current_session in tqdm_gen_session:
            args.count_prompt, args.count_img = 0, 0
            tqdm_gen_session.set_postfix({'Current session': current_session})
            # load prototypes and prompts up to now
            up2now_classes = args.class_seq[:args.fg_nc + current_session * args.task_size]
            # set visible prototypes for current session
            if args.use_mc_proto:
                k_mc_id = [idx for idx, value in enumerate(k_mc_mapping) if value.item() in up2now_classes]
                v_mc_id = [idx for idx, value in enumerate(v_mc_mapping) if value.item() in up2now_classes]
                k_visible_protos = k_mc_protos[k_mc_id, :]
                v_visible_protos = v_mc_protos[v_mc_id, :]
            else:
                k_visible_protos = k_protos[up2now_classes, :]
                v_visible_protos = v_protos[up2now_classes, :]
            # set visible prompts
            visible_prompts = prompts[up2now_classes, :]
            assert prompter.prompts.size() == visible_prompts[0, :].size()
            # set current session
            cl_helper.set_current_session(current_session)
            cl_helper_prompt.set_current_session(current_session)
            # loop
            tqdm_gen_task = tqdm.tqdm(range(current_session+1)) 
            tqdm_gen_task.set_description('Task Loop')
            for current_task in tqdm_gen_task:
                tqdm_gen_task.set_postfix({'Current task':current_task})
                # set current task
                cl_helper.set_current_task(current_task)
                cl_helper_prompt.set_current_task(current_task)

                # get current classes
                if current_task == 0:
                    classes = up2now_classes[:args.fg_nc]
                else:
                    s_idx = args.fg_nc + (current_task - 1) * args.task_size
                    e_idx = args.fg_nc + current_task * args.task_size
                    classes = up2now_classes[s_idx:e_idx]

                test_dataset.set_classes(classes)
                test_loader = DataLoader(dataset=test_dataset,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         batch_size=args.bs,
                                         drop_last=False)

                for _, (imgs, labels) in enumerate(test_loader):
                    imgs, labels = imgs.to(args.device), labels.to(args.device)
                    output = feature_extractor(imgs)
                    # measure distances
                    distances = utils.measure_distance(args.similarity, output, k_visible_protos)
                    # calculate accuracy
                    if args.use_mc_proto:
                        (top1_count, topk_count), (topk_indexes, topk_labels, top1_predicts) = \
                        utils.cal_acc_mc(args, distances, labels, k_mc_id, k_mc_mapping)
                    else:
                        (top1_count, topk_count), (topk_indexes, topk_labels, top1_predicts) = \
                        utils.cal_acc(args, distances, labels, up2now_classes)
                    # update metric
                    cl_helper.update(top1_count, topk_count, len(labels))
                    cl_helper.update_raw(top1_predicts, labels)
                    # prompt 
                    if args.use_mc_proto:
                        top1_count_prompt, top1_predicts_prompt = \
                        prompt_for_mc_prototype(args, imgs, labels, prompter, prompts, topk_labels, \
                                                v_visible_protos, v_mc_id, v_mc_mapping, v_mc_protos)
                    else:
                        top1_count_prompt, top1_predicts_prompt = \
                        prompt_for_prototype(args, imgs, labels, prompter, \
                                            v_visible_protos, visible_prompts, up2now_classes, topk_indexes)
                    # update
                    cl_helper_prompt.update(top1_count_prompt, 0, len(labels))
                    cl_helper_prompt.update_raw(top1_predicts_prompt, labels)

            count_img_collection.append(args.count_img)
            count_prompt_collection.append(args.count_prompt)

        cl_helper.get_result_report()
        pprint.pprint(cl_helper.result_report, sort_dicts=False)
        args.save_helper.save_txt(cl_helper.result_report, 'result')
        # save raw result
        args.save_helper.save_torch(cl_helper.raw_result_all_sessions, 'raw_result')

        cl_helper_prompt.get_result_report()
        # calculte number of retrived prompt
        count_img_collection = np.array(count_img_collection)
        count_prompt_collection = np.array(count_prompt_collection)
        session_wise_count = count_prompt_collection / count_img_collection
        total_avg_count = np.sum(count_prompt_collection) / np.sum(count_img_collection)

        cl_helper_prompt.result_report['Count'] = {'session_wise': session_wise_count, 'total_avg': total_avg_count}
        pprint.pprint(cl_helper_prompt.result_report, sort_dicts=False)
        args.save_helper.save_txt(cl_helper_prompt.result_report, 'result_prompt')
        # save raw result
        args.save_helper.save_torch(cl_helper_prompt.raw_result_all_sessions, 'raw_result_prompt')

    
def prompt_for_prototype(args, imgs, gt_labels, prompter, visible_proto, 
                         visible_prompts, up2now_classes, topk_indexes):
    top1_count = 0
    predict_labels = []
    for bs_idx, _ in enumerate(gt_labels):  # ugly nested loops
        cur_features = []
        for topk_idx in topk_indexes[bs_idx]:
            prompter.prompts = nn.Parameter(visible_prompts[topk_idx, :])  # insert corresponding prompt
            output = prompter(imgs[bs_idx, :].unsqueeze(0))
            cur_features.append(output)
        cur_features = torch.stack(cur_features, dim=0).squeeze()  # shape: (topk, embed_dim)
        topk_prototypes = torch.index_select(visible_proto, 0, topk_indexes[bs_idx])  # shape: (topk, embed_dim)
        distances = utils.measure_distance(args.similarity, cur_features, topk_prototypes)  # shape: (topk, topk)
        index = torch.argmin(distances)
        predict_label = up2now_classes[topk_indexes[bs_idx][index % distances.size(-1)]]
        predict_labels.append(torch.tensor(predict_label))
        if predict_label == gt_labels[bs_idx]:
            top1_count += 1
    predict_labels = torch.stack(predict_labels, dim=0).squeeze()
    return top1_count, predict_labels

    
def prompt_for_mc_prototype(args, imgs, gt_labels, prompter, prompts, \
                            topk_labels, visible_mc_proto, v_mc_id, \
                            mc_proto_mapping=None, mc_proto=None):
    
    assert len(v_mc_id) == len(visible_mc_proto)
    # class_session map
    class_session_map = args.class_session_map
    # infer with prompt
    top1_count = 0
    predict_labels = []
    for bs_idx, _ in enumerate(gt_labels):  # ugly nested loops
        args.count_img += 1
        session_result = {}
        cur_features = []
        unique_class = torch.unique(topk_labels[bs_idx,:])
        for class_idx in unique_class:
            class_idx = class_idx.item()
            if class_session_map[class_idx] not in session_result.keys():
                prompter.prompts = nn.Parameter(prompts[class_idx, :])  # insert corresponding prompt
                output = prompter(imgs[bs_idx, :].unsqueeze(0))
                args.count_prompt += 1
                session_result[class_session_map[class_idx]] = output
            else:
                output = session_result[class_session_map[class_idx]]
            cur_features.append(output)
        cur_features = torch.cat(cur_features, dim=0)  # shape: (topk, embed_dim)
        # get label   
        vicinity_protos = visible_mc_proto
        vicinity_labels = mc_proto_mapping[v_mc_id]
        distances = utils.measure_distance(args.similarity, cur_features, vicinity_protos)  # shape: (unique_class, topk)
        index = torch.argmin(distances)
        predict_label = vicinity_labels[index % distances.size(-1)]
        predict_labels.append(predict_label)
        # count correct
        if predict_label == gt_labels[bs_idx]:
            top1_count += 1
    predict_labels = torch.stack(predict_labels, dim=0).squeeze()
    return top1_count, predict_labels