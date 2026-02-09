import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks




class AdaAttNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--image_encoder_path', required=True, help='path to pretrained image encoder')
        parser.add_argument('--skip_connection_3', action='store_true',
                            help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--shallow_layer', action='store_true',
                            help='if specified, also use features of shallow layers')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=0., help='weight for L2 content loss')
            parser.add_argument('--lambda_global', type=float, default=10., help='weight for L2 style loss')
            parser.add_argument('--lambda_local', type=float, default=3.,
                                help='weight for attention weighted style loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        # 定义模型可视化图像名称
        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder', 'transformer']
        parameters = []
        self.max_sample = 64 * 64  # 最大采样数量

        # 创建核心网络模块
        if opt.skip_connection_3:  # 如果为True就在Relu-3后面添加跳跃连接，增强模型的特征融合能力
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                         max_sample=self.max_sample)  # 不同尺度的通道级联，并且大的尺寸同一采样到64*64
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)  # 初始化网络
            self.model_names.append('adaattn_3')  # 应该也是将学习到的参数放进去
            parameters.append(self.net_adaattn_3.parameters())
        if opt.shallow_layer:
            channels = 512 + 256 + 128 + 64
        else:
            channels = 512
        transformer = networks.Transformer(
            in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)  # key_planes=第五层+第四层+第三层+第二层
        decoder = networks.Decoder(opt.skip_connection_3)  # 创建decoder实例，
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transformer.parameters())
        # --- 新增代码开始 (这部分和之前的回答一样) ---

        # 如果使用浅层特征，通道数是各层之和；否则是单层的通道数
        dim_3 =  256   #448 if opt.shallow_layer else
        dim_4 = 512  # 448 + 512 = 960   960 if opt.shallow_layer else
        dim_5 = 512  # 960 + 512 = 1472 1472 if opt.shallow_layer else
        # 定义 DilatedMDTA 模块
        if opt.skip_connection_3:
            style_attn_3 = networks.DilatedMDTA(dim=dim_3, num_heads=8, bias=False)
            self.net_style_attn_3 = networks.init_net(style_attn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('style_attn_3')
            parameters.append(self.net_style_attn_3.parameters())

        style_attn_4 = networks.DilatedMDTA(dim=dim_4 , num_heads=8,
                                            bias=False)
        self.net_style_attn_4 = networks.init_net(style_attn_4, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.model_names.append('style_attn_4')
        parameters.append(self.net_style_attn_4.parameters())

        style_attn_5 = networks.DilatedMDTA(dim=dim_5, num_heads=8,
                                            bias=False)
        self.net_style_attn_5 = networks.init_net(style_attn_5, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.model_names.append('style_attn_5')
        parameters.append(self.net_style_attn_5.parameters())
        # --- 新增代码结束 ---

#==========================================content-enhance相关代码增加========================================
        # ==================== 修改开始 ====================

        # --- 创建核心网络模块 ---
        # 根据 shallow_layer 选项确定 AFE 模块的输入维度
        dim_k2, dim_k3, dim_k4 = 256, 512, 512
        key_planes_transformer = 512
        key_planes_adaattn3 = 256

        if opt.shallow_layer:
            dim_k2 += 128 + 64  # 448
            dim_k3 += 256 + 128 + 64  # 960
            dim_k4 += 512 + 256 + 128 + 64  # 1472
            key_planes_transformer = dim_k3  # 或者 dim_k4，取决于Transformer的设计
            key_planes_adaattn3 = dim_k2

        # 实例化 AFE 模块?///////二次添加====================================
        afe_2 = networks.AFE(dim=256)
        afe_3 = networks.AFE(dim=512)
        afe_4 = networks.AFE(dim=512)
        self.net_afe_2 = networks.init_net(afe_2, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_afe_3 = networks.init_net(afe_3, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_afe_4 = networks.init_net(afe_4, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.model_names.append('afe_2')
        self.model_names.append('afe_3')
        self.model_names.append('afe_4')
        parameters.append(self.net_afe_2.parameters())
        parameters.append(self.net_afe_3.parameters())
        parameters.append(self.net_afe_4.parameters())
        # 实例化 AFE 模块?///////二次添加====================================


        afe_k2 = networks.AFE(dim=dim_k2)
        afe_k3 = networks.AFE(dim=dim_k3)
        afe_k4 = networks.AFE(dim=dim_k4)
        self.net_afe_k2 = networks.init_net(afe_k2, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_afe_k3 = networks.init_net(afe_k3, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_afe_k4 = networks.init_net(afe_k4, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.model_names.append('afe_k2')
        self.model_names.append('afe_k3')
        self.model_names.append('afe_k4')
        parameters.append(self.net_afe_k2.parameters())
        parameters.append(self.net_afe_k3.parameters())
        parameters.append(self.net_afe_k4.parameters())

        if opt.skip_connection_3:  # 如果为True就在Relu-3后面添加跳跃连接，增强模型的特征融合能力
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=dim_k2, max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)  # 初始化网络
            self.model_names.append('adaattn_3')  # 应该也是将学习到的参数放进去
            parameters.append(self.net_adaattn_3.parameters())

        # Transformer的key_planes现在由AFE的输出维度决定
        transformer = networks.Transformer(
            in_planes=512, key_planes=dim_k3, shallow_layer=opt.shallow_layer)  # 使用 dim_k3 作为示例
        decoder = networks.Decoder(opt.skip_connection_3)  # 创建decoder实例
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transformer.parameters())

        # ==================== 修改结束 ====================
# ==========================================content-enhance相关代码增加========================================

        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        self.criterionMSE = torch.nn.MSELoss().to(self.device)
        if self.isTrain:
            self.loss_names = ['content', 'global', 'local']

            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_global = torch.tensor(0., device=self.device)
            self.loss_local = torch.tensor(0., device=self.device)
            self.loss_content = torch.tensor(0., device=self.device)

    def set_input(self, input_dict):
        self.c = input_dict['c'].to(self.device)
        self.s = input_dict['s'].to(self.device)
        self.image_paths = input_dict['name']

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape  # 获得当前层也就是尺寸最小层的特征，用于下面的尺度裁剪
            for i in range(last_layer_idx):  # 0-last_layer_idx-1
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
                # 将result列表中添加尺寸裁剪后的特征，这些特征并且经过了均值方差的标准化
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))  # 将当前层特征同样添加到result中
            return torch.cat(results, dim=1)  # 进行在通道维度上的拼接
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    def forward(self):
        self.c_feats = self.encode_with_intermediate(self.c)
        self.s_feats = self.encode_with_intermediate(self.s)
        # ## 修改点 ##
        # 1. 获取原始的 content_key 和 style_key
        # c_key_3 = self.get_key(self.c_feats, 2, self.opt.shallow_layer)
        # s_key_3 = self.get_key(self.s_feats, 2, self.opt.shallow_layer)
        # c_key_4 = self.get_key(self.c_feats, 3, self.opt.shallow_layer)
        # s_key_4 = self.get_key(self.s_feats, 3, self.opt.shallow_layer)
        # c_key_5 = self.get_key(self.c_feats, 4, self.opt.shallow_layer)
        # s_key_5 = self.get_key(self.s_feats, 4, self.opt.shallow_layer)
        #
        # # 2. 对 style_key 应用通道注意力，生成 attended_style_key
        # if self.opt.skip_connection_3:
        #     attended_s_key_3 = self.net_style_attn_3(s_key_3)
        # attended_s_key_4 = self.net_style_attn_4(s_key_4)
        # attended_s_key_5 = self.net_style_attn_5(s_key_5)
        # # 3. 将 attended_style_key 传入网络，但原始的 style_feats 保持不变
        # if self.opt.skip_connection_3:
        #     c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2], c_key_3,
        #                                         attended_s_key_3, self.seed)
        # else:
        #     c_adain_feat_3 = None
        # cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
        #                           c_key_4, s_key_4,  # 这里的 s_key_4 保持原样，也可以换成 attended_s_key_4 进行实验
        #                           c_key_5, s_key_5,  # 这里的 s_key_5 保持原样，也可以换成 attended_s_key_5 进行实验
        #                           attended_s_key_4,  # 传入新的 attended key
        #                           attended_s_key_5,  # 传入新的 attended key
        #                           self.seed)
        s_att_3=self.s_feats[2]
        s_att_4 = self.s_feats[3]
        s_att_5 = self.s_feats[4]  #Vgg中得来的特征

        s_feats_att3 = self.net_style_attn_3(s_att_3)
        s_feats_att4 = self.net_style_attn_4(s_att_4)
        s_feats_att5 = self.net_style_attn_5(s_att_5)  #这几个特征经过DilatedMDTA强化，未来的V


        c_feats_k2 = self.net_afe_k2(self.get_key(self.c_feats, 2, self.opt.shallow_layer))
        s_feats_k2 = self.net_afe_k2(self.get_key(self.s_feats, 2, self.opt.shallow_layer))
        c_feats_k3 = self.net_afe_k3(self.get_key(self.c_feats, 3, self.opt.shallow_layer))
        s_feats_k3 = self.net_afe_k3(self.get_key(self.s_feats, 3, self.opt.shallow_layer))
        c_feats_k4 = self.net_afe_k4(self.get_key(self.c_feats, 4, self.opt.shallow_layer))
        s_feats_k4 = self.net_afe_k4(self.get_key(self.s_feats, 4, self.opt.shallow_layer))
        #后三层特征先拼接然后 通过AFE进行强化，作为骨干，产生softmax矩阵那两哥们

        c_feats_2=self.net_afe_2(self.c_feats[2])
        c_feats_3=self.net_afe_3(self.c_feats[3])
        c_feats_4=self.net_afe_4(self.c_feats[4])#内容特征通过AFE强化了一次未来的主干信息



        if self.opt.skip_connection_3:
            # inplane=256，keyplane=256+128+64
            c_adain_feat_3 = self.net_adaattn_3(c_feats_2, s_feats_att3,
                                                c_feats_k2,
                                                s_feats_k2, self.seed)
        else:
            c_adain_feat_3 = None
        cs = self.net_transformer(c_feats_3, s_feats_att4, c_feats_4, s_feats_att5,
                                  c_feats_k3,
                                  s_feats_k3,
                                  c_feats_k4,
                                  s_feats_k4, self.seed)
        self.cs = self.net_decoder(cs, c_adain_feat_3)



        """ if self.opt.skip_connection_3:
            # inplane=256，keyplane=256+128+64
            c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2],
                                                self.get_key(self.c_feats, 2, self.opt.shallow_layer),
                                                self.get_key(self.s_feats, 2, self.opt.shallow_layer), self.seed)
        else:
            c_adain_feat_3 = None
        cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
                                  self.get_key(self.c_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.c_feats, 4, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 4, self.opt.shallow_layer), self.seed)
        self.cs = self.net_decoder(cs, c_adain_feat_3)"""


    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[i]),
                                                       networks.mean_variance_norm(self.c_feats[i]))

    # 测试loss ===============================================================================================
    def content_loss(self,stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[-1])
                                               ,networks.mean_variance_norm(self.c_feats[-1]))+ self.criterionMSE(
            networks.mean_variance_norm(stylized_feats[-2]),networks.mean_variance_norm(self.c_feats[-2]))

    # def content_loss(self, stylized_feats):
    #     self.loss_content = torch.tensor(0., device=self.device)
    #     self.loss_content += self.criterionMSE(stylized_feats[-1], self.c_feats[-1]) + self.criterionMSE(
    #         stylized_feats[-2], self.c_feats[-2])

    def style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        # if self.opt.lambda_global > 0:
        for i in range(2, 5):
            s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
            stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
            self.loss_global += self.criterionMSE(
                stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)

# 测试loss ===============================================================================================

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.opt.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
        self.loss_local = torch.tensor(0., device=self.device)
        if self.opt.lambda_local > 0:
            for i in range(1, 5):
                c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i],
                                                     std * networks.mean_variance_norm(self.c_feats[i]) + mean)

    def compute_losses(self):
        stylized_feats = self.encode_with_intermediate(self.cs)
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.loss_content = self.loss_content * self.opt.lambda_content
        self.loss_local = self.loss_local * self.opt.lambda_local
        self.loss_global = self.loss_global * self.opt.lambda_global

    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_global + self.loss_local
        loss.backward()
        self.optimizer_g.step()


#     =====================================修改1 begin GSA=================================================

