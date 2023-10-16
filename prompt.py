import utils
import torch
import torch.nn as nn


class ProTLearner(nn.Module):
    def __init__(self, args, vit_model, prev_prompts=None):
        super().__init__()
        # learnable prompt
        self.args = args
        self.l_p = args.l_p  # length of prompt
        self.d_p = args.d_p  # deep prompt
        self.prompt_idx = 0
        self.vit_model = vit_model

        if self.d_p: 
            self.n_p = len(vit_model.blocks)  # number of the prompts
            self.insert_position = list(range(1, len(vit_model.blocks)))
        else:
            self.n_p = 1
            self.insert_position = []

        # initialize prompt
        if prev_prompts is None:
            self.prompts = nn.Parameter(torch.zeros(self.n_p, self.l_p, self.vit_model.embed_dim))
        else:
            self.prompts = nn.Parameter(prev_prompts)
        self.register_parameter(name='prompts', param=self.prompts)

        if not self.args.finetune_vit:
            # only update prompt
            self._set_only_prompt_trainable()


    def _set_only_prompt_trainable(self):
        # set all parameters except prompt fixed
        for name, param in self.named_parameters():
            if "prompts" not in name:
                param.requires_grad_(False)
           

    def get_prompt_param(self):
        trainable_parameter = []
        # set all parameters except prompt fixed
        for _, param in self.named_parameters():
            if param.requires_grad:
                trainable_parameter.append(param)
        return trainable_parameter          


    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.vit_model.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.vit_model.cls_token.expand(B, -1, -1)

        # add cls token
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        try:
            pos_embed = self.vit_model.interpolate_pos_encoding(x, w, h)
        except:
            pos_embed = self.vit_model.pos_embed

        x = x + pos_embed

        # insert prompt token
        prompt = self.prompts[self.prompt_idx].unsqueeze(0).expand(B, -1, -1)
        cls_prompt = torch.cat((x[:,0,:].unsqueeze(1), prompt), dim=1)
        x = torch.cat((cls_prompt, x[:,1:,:]), dim=1)
        self.prompt_idx += 1

        return self.vit_model.pos_drop(x)

    def forward(self, x):
        # reset
        init_flag = True
        self.prompt_idx = 0
        # forward
        x = self.prepare_tokens(x)
        for idx, blk in enumerate(self.vit_model.blocks):
            if idx in self.insert_position:
                if self.d_p:
                    x[:,1:self.args.l_p+1,:] = self.prompts[self.prompt_idx,:]
                else:
                    if init_flag:
                        prompt = self.prompts[self.prompt_idx,:].unsqueeze(0).expand(x.size(0), -1, -1)
                        cls_prompt = torch.cat((x[:,0,:].unsqueeze(1), prompt), dim=1)
                        x = torch.cat((cls_prompt, x[:,1:,:]), dim=1)
                    else:
                        x[:,1:self.args.l_p+1,:] = self.prompts[self.prompt_idx,:]
                    init_flag = False
                self.prompt_idx += 1
            x = blk(x)
            
        if self.args.pretrain_method == 'mae':
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.vit_model.fc_norm(x)
            return outcome
        else:
            x = self.vit_model.norm(x)
            return x[:, 0]


class PromptHead(nn.Module):
    def __init__(self, args, vit_embed_dim):
        super().__init__()
        self.args = args
        self.vit_embed_dim = vit_embed_dim
        self.mlp = None
        self.linear_cls = None
        if args.add_mlp:
            self.set_mlp_neck()
        if args.add_cls:
            self.set_linear_cls()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_mlp_neck(self):
        nlayers = self.args.mlp_layer_num
        bottleneck_dim = self.vit_embed_dim
        hidden_dim = self.args.mlp_hidden_dim
        use_bn = self.args.mlp_use_bn
        in_dim = self.vit_embed_dim
        
        if nlayers == 0:
            self.mlp = nn.Identity()
        else:
            nlayers = max(nlayers, 1)
            if nlayers == 1:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                layers = [nn.Linear(in_dim, hidden_dim)]
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
                for _ in range(nlayers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    if use_bn:
                        layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.GELU())
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
                self.mlp = nn.Sequential(*layers)
            self.apply(self._init_weights)

    def set_linear_cls(self):
        self.linear_cls = nn.utils.weight_norm(nn.Linear(self.vit_embed_dim, self.args.cls_dim, bias=False))
        self.linear_cls.weight_g.data.fill_(1)

    def get_head_parameters(self):
        param = []
        if self.mlp:
            param += self.mlp.parameters()
        if self.linear_cls:
            param += self.linear_cls.parameters()
        return param

    def forward(self, x):
        if self.mlp:
            x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        neck_output = x
        if self.linear_cls:
            x = self.linear_cls(x)
        return x, neck_output