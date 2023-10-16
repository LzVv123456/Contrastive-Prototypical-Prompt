import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
import utils
import copy
import losses
from tqdm import tqdm
import prototype as prot
from datasets import ProtoDataset
from prompt import ProTLearner, PromptHead


class Trainer(object):
    def __init__(self, args, vit_model, train_dataset, gen_proto_dataset):
        super().__init__()
        self.args = args
        self.vit_model = vit_model
        self.dataset = train_dataset
        self.gen_proto_dataset = gen_proto_dataset

        self.proto = []
        self.proto_var = []
        self.mc_proto = []
        self.mc_proto_var = []
        self.prototype_prompts = []
        self.prompter = None
        self.head = None
        self.data_loader = None


    def get_optimizer(self, optm, trainable_parameter):
        if optm == "adamw":
            optimizer = torch.optim.AdamW(trainable_parameter)  # to use with ViTs
        elif optm == "sgd":
            optimizer = torch.optim.SGD(trainable_parameter, lr=0, momentum=0.9)  # lr and wd is set by scheduler
        elif optm == "adam":
            optimizer = torch.optim.Adam(trainable_parameter)  # to use with ViTs
        else:
            raise NotImplementedError
        return optimizer


    def init_loss_func(self, classes_up2now):
        if self.args.loss == 'protocon':
            if self.args.use_mc_proto:
                self.loss_function = losses.ProtoConLoss(self.args, self.mc_proto, classes_up2now, self.proto_dataloader)
            else:
                self.loss_function = losses.ProtoConLoss(self.args, self.proto, classes_up2now, self.proto_dataloader)
        elif self.args.loss == 'supcon':
            self.loss_function = losses.SupConLoss(self.args)
        elif self.args.loss == 'ce':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError


    def get_loss(self, output, mlp_output, target, cur_classes, ):
        if self.args.loss == 'ce':
            if self.args.cls_dim == self.args.task_size: # relabel
                new_target = torch.zeros(target.size())
                for idx, label in enumerate(cur_classes):
                    new_target[target == label] = idx
                target = new_target.cuda().long()
            loss = self.loss_function(output, target)
        else:
            if self.args.loss == 'supcon':
                mlp_output = mlp_output.unsqueeze(dim=1)
            loss = self.loss_function(mlp_output, target)
        return loss


    def train_discriminate(self):
        self.prompter = ProTLearner(self.args, self.vit_model)
        self.prompter = self.prompter.cuda()
        
        # start session loop
        for current_session in range(self.args.task_num+1):
            # init classes
            if current_session == 0:
                classes = self.args.class_seq[:self.args.fg_nc]
                classes_up2now = None
            else:
                s_idx = self.args.fg_nc + (current_session - 1) * self.args.task_size
                e_idx = self.args.fg_nc + current_session * self.args.task_size
                classes = self.args.class_seq[s_idx:e_idx]
                classes_up2now = self.args.class_seq[:s_idx]
            print('current session id: {}'.format(current_session))

            # set prototype augmentation dataset
            if classes_up2now is not None:  
                if self.args.use_mc_proto:
                    self.proto_dataset = ProtoDataset(self.args, [self.mc_proto[idx] for idx in classes_up2now], \
                                                      [self.mc_proto_var[idx] for idx in classes_up2now], classes_up2now)
                else:
                    self.proto_dataset = ProtoDataset(self.args, [self.proto[idx] for idx in classes_up2now], \
                                                      [self.proto_var[idx] for idx in classes_up2now], classes_up2now)

                proto_sampler = RandomSampler(self.proto_dataset, replacement=True, num_samples=self.args.proto_aug_bs)
                self.proto_dataloader = DataLoader(dataset=self.proto_dataset,
                                                    num_workers=0,
                                                    batch_size=self.args.proto_aug_bs,
                                                    sampler=proto_sampler,
                                                    shuffle=False,
                                                    drop_last=False)
            else:
                self.proto_dataloader = None

            # set loss function
            self.init_loss_func(classes_up2now)

            # set current dataset
            self.dataset.set_classes(classes)
            # get current datasampler and dataloader
            self.data_loader = DataLoader(dataset=self.dataset,
                                          num_workers=self.args.num_workers,
                                          batch_size=self.args.bs,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=True)

            # init current prompter
            if self.prompter is not None and self.args.init_with_prev_prompt:
                prev_prompts = self.prompter.prompts.detach()
            else:
                prev_prompts = None  
            self.prompter = ProTLearner(self.args, self.vit_model, prev_prompts)
            self.prompter = self.prompter.cuda()
            prompt_param = self.prompter.get_prompt_param()

            # init_head
            if self.args.loss == 'ce' and self.args.incre_cls:
                self.args.cls_dim = (current_session + 1) * self.args.task_size
            self.head = PromptHead(self.args, self.prompter.vit_model.embed_dim).cuda()
            head_param = self.head.get_head_parameters()

            # set optimizers
            self.p_optm = self.get_optimizer(self.args.p_optm, prompt_param)
            if head_param:
                self.h_optm = self.get_optimizer(self.args.h_optm, head_param)
            else:
                self.h_optm = None

            # set scheduler
            self.p_lr_scheduler = utils.cosine_scheduler(
                self.args.p_lr,
                self.args.p_min_lr,
                self.args.epochs, len(self.data_loader),
                warmup_epochs=min(self.args.warmup_epochs, int(self.args.epochs*0.1)),
            )
            self.p_wd_schedule = utils.cosine_scheduler(
                self.args.p_wd,
                self.args.p_wd_end,
                self.args.epochs, len(self.data_loader),
            )
            self.h_lr_scheduler = utils.cosine_scheduler(
                self.args.h_lr,
                self.args.h_min_lr,
                self.args.epochs, len(self.data_loader),
                warmup_epochs=min(self.args.warmup_epochs, int(self.args.epochs*0.1)),
            )
            self.h_wd_schedule = utils.cosine_scheduler(
                self.args.h_wd,
                self.args.h_wd_end,
                self.args.epochs, len(self.data_loader),
            )

            # train loop
            tqdm_gen = tqdm(range(self.args.epochs))
            for epoch in tqdm_gen:
                epoch_loss = 0
                self.prompter.train()
                self.head.train()
                for step, (imgs, labels) in enumerate(self.data_loader):
                    step = len(self.data_loader) * epoch + step
                    # adjust lr and wd
                    for _, param_group in enumerate(self.p_optm.param_groups):
                        param_group["lr"] = self.p_lr_scheduler[step]
                        param_group["weight_decay"] = self.p_wd_schedule[step]
                    if self.h_optm:
                        for _, param_group in enumerate(self.h_optm.param_groups):
                            param_group["lr"] = self.h_lr_scheduler[step]
                            param_group["weight_decay"] = self.h_wd_schedule[step]

                    # forward 
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                    logits = self.prompter(imgs)  
                    output, mlp_output = self.head(logits)
                    # get loss
                    loss = self.get_loss(output, mlp_output, labels, classes)
                    # optimize
                    self.p_optm.zero_grad()
                    if self.h_optm:
                        self.h_optm.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.prompter.parameters(), 5.0)
                    nn.utils.clip_grad_norm_(self.head.parameters(), 5.0)
                    self.p_optm.step()
                    if self.h_optm:
                        self.h_optm.step()
                    # add loss
                    epoch_loss += loss.item()

                epoch_loss /= len(self.data_loader)
                tqdm_gen.set_postfix({'epoch avg loss': epoch_loss})
                self.args.save_helper.tb.add_scalar('Task:{}, Epoch/Loss'.format(current_session), epoch_loss, epoch)

            # update prototypes
            for proto_id in classes:
                self.prototype_prompts.append(copy.deepcopy(self.prompter.prompts))  
                # generate value prototyes
                mean, var, mc_proto, mc_proto_var = \
                self.generate_proto_relatives(self.prompter, proto_id)
                self.proto.append(mean.detach())
                self.proto_var.append(var.detach())
                self.mc_proto.append(mc_proto.detach())
                self.mc_proto_var.append(mc_proto_var.detach())

        # close tb
        self.args.save_helper.tb.close()
        # resort according to ascending order
        self.prototype_prompts = utils.adjust_order(self.prototype_prompts, self.args.class_seq)
        # stack all prototype prompts
        self.prototype_prompts = torch.stack(self.prototype_prompts, dim=0)
        return self.prototype_prompts


    def generate_proto_relatives(self, feature_extractor, class_id):
        with torch.no_grad():
            feature_extractor.eval()
            features, var, mean, _ = \
            prot.gen_single_proto(self.args, feature_extractor, \
                                  class_id, self.gen_proto_dataset)
            mc_proto, mc_proto_var, _, _ = prot.gen_single_mc_proto(self.args, features)
            mc_proto = mc_proto.cuda()
            mc_proto_var = mc_proto_var.cuda()
        return mean, var, mc_proto, mc_proto_var
