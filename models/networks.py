import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)#此处通过函数得到了均值和方差两者形状都为B C 1 1
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)#所以此处要使用mean.expand,相当于广播到B C H W
    return normalized_feat


class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        #新增relu激活
        self.relu = nn.ReLU()
    #新增一个可学习的参数‘W',包含两个值，分别对应两个分支权重
    #用nn.parameter封装，模型训练时会自动更新它
        #self.w=nn.parameter(torch.one(2))
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)#q  content_key和style_key都是只剩下内容的框架，去掉了均值方差等通道信息
        G = self.g(style_key)#k
        H = self.h(style)#v
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:#判断是否超过数值
            if seed is not None:
                torch.manual_seed(seed)#弄了这个，随机数就能返回，在compute_losses中，为了计算local_feature_loss，也需要复现这个采样过程，因此固定的种子是必须的。
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)#B C HW->B HW C
     # --- 核心改动：自适应稀疏注意力的计算 ---====================================================
        # 1. 计算原始(未激活的)注意力得分
        raw_attn = torch.bmm(F, G)  # B, HW_content, HW_style

        # 2. 计算双分支注意力图
        # 分支一：密集注意力 (Dense Attention)，使用Softmax
        attn_dense = self.sm(raw_attn)
        # 分支二：稀疏注意力 (Sparse Attention)，使用ReLU的平方
        attn_sparse = self.sm(self.relu(raw_attn))

        # 3. 计算自适应权重 (手动实现Softmax以确保权重和为1)
        w_sum = torch.sum(torch.exp(self.w))
        w_dense = torch.exp(self.w[0]) / w_sum
        w_sparse = torch.exp(self.w[1]) / w_sum

        # 4. 加权融合两个分支的注意力图，得到最终的注意力矩阵 S
        S = attn_dense * w_dense + attn_sparse * w_sparse
    # --- 核心改动结束 ---=======================================================================
        """S = torch.bmm(F, G)#B HW HW 逐点查询"""
        # S: b, n_c, n_s
        S = self.sm(S)#此处是一个注意力矩阵
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):#in_planes512,key_planes512+256+128+64
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
                                        key_planes=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):#后加了attended_style4_1_key, attended_style5_1_key,--->又去掉了
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed) +
            self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed))))#style4_1_key被上述替代


class Decoder(nn.Module):

    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs

# class DilatedMDTA(nn.Module):
#     """
#     实现 Dilated Grouped Channel-Wise Self-Attention (论文中的 Dilated G-CSA)。
#     该模块是 DTAB 的一部分，负责全局信息交互。
#     """
#
#     def __init__(self, dim, num_heads=8, bias=False):
#         """
#         初始化函数。
#
#         Args:
#             dim (int): 输入和输出的特征维度 (通道数)。
#             num_heads (int): 注意力头的数量。这对应论文中的“分组”。
#             bias (bool): 卷积层是否使用偏置。
#         """
#         super(DilatedMDTA, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.eps = 1e-6#-------------------------->尝试防止黄图
#
#         # 1x1 卷积生成 Q, K, V
#         self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#
#         # 关键的扩张深度可分离卷积，用于在生成 Q, K, V 时维持盲点特性
#         # The key dilated depth-wise convolution to maintain the blind-spot property when generating Q, K, V
#         self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 3,
#                                     bias=bias)
#
#         # 1x1 卷积用于输出
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         #x=mean_variance_norm(x)
#         b, c, h, w = x.shape
#
#         # 通过扩张卷积生成 q, k, v
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q, k, v = qkv.chunk(3, dim=1)
#
#         # 重排形状以进行多头注意力计算
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         # 归一化
#         # q = torch.nn.functional.normalize(q, dim=-1)
#         # k = torch.nn.functional.normalize(k, dim=-1)
#         # 归一化 q 和 k，并增加 eps 防止除以零
#         q = q / (torch.norm(q, dim=-1, keepdim=True) )
#         k = k / (torch.norm(k, dim=-1, keepdim=True) )
#
#         # 计算注意力矩阵
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         # 应用注意力权重
#         out = (attn @ v)
#
#         # 重排回原始图像格式
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out
#     =====================================修改1endingGSA=================================================


class DilatedMDTA(nn.Module):
    """
    为强化风格表征而修改的 DilatedMDTA。
    此版本严格遵循 Style_SA 的编码风格：
    1. 使用 torch.bmm 进行矩阵乘法。
    2. 使用 view 和 permute 进行张量变形。
    3. 移除了 QK 归一化以保留风格模长信息。
    4. 采用 f, g, h 独立网络路径。
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super(DilatedMDTA, self).__init__()#StyleDilatedMDTA_BMM
        self.num_heads = num_heads

        # 可学习的温度参数，用于缩放注意力分数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 使用 f, g, h 独立路径来模拟 Style_SA 的结构
        # 每个路径包含一个 1x1 卷积和一个扩张深度卷积
        self.f = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=0, groups=dim, bias=bias)
        )
        self.g = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=0, groups=dim, bias=bias)
        )
        self.h = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=0, groups=dim, bias=bias)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, style_feat):
        B, C, H, W = style_feat.shape
        C_per_head = C // self.num_heads

        # 1. 通过 f, g, h 网络生成 Q, K, V 特征
        f_x = self.f(style_feat)
        g_x = self.g(style_feat)
        h_x = self.h(style_feat)

        # 2. Reshape Q, K for bmm (实现分组通道注意力)
        # q: (B, C, H, W) -> (B*num_heads, C_per_head, H*W)
        q_bmm = f_x.view(B * self.num_heads, C_per_head, H * W)

        # k: (B, C, H, W) -> view -> (B, num_heads, C_ph, H*W) -> permute -> (B, num_heads, H*W, C_ph) -> view -> (B*h, H*W, C_ph)
        k_bmm = g_x.view(B, self.num_heads, C_per_head, H * W).permute(0, 1, 3, 2).contiguous().view(B * self.num_heads,
                                                                                                     H * W, C_per_head)

        # 3. 计算 Energy (未归一化的注意力分数)
        # energy: (B*h, C_ph, H*W) @ (B*h, H*W, C_ph) -> (B*h, C_ph, C_ph)
        energy = torch.bmm(q_bmm, k_bmm)

        # 4. 应用温度参数并计算 Attention
        # 将 temperature 从 (num_heads, 1, 1) 扩展到 (B*num_heads, 1, 1) 以进行批处理
        temp = self.temperature.repeat(B, 1, 1)
        attention = self.softmax(energy * temp)

        # 5. Reshape V 并应用 Attention
        # v: (B, C, H, W) -> (B*num_heads, C_per_head, H*W)
        v_bmm = h_x.view(B * self.num_heads, C_per_head, H * W)

        # out: (B*h, C_ph, C_ph) @ (B*h, C_ph, H*W) -> (B*h, C_ph, H*W)
        out_bmm = torch.bmm(attention, v_bmm)

        # 6. 将输出 Reshape 回原始图像格式
        # (B*h, C_ph, H*W) -> (B, C, H, W)
        out = out_bmm.view(B, C, H * W).view(B, C, H, W)

        # 7. 最终的卷积和残差连接
        out = self.out_conv(out)
        out += style_feat  # 加上输入的特征，形成残差结构

        return out



#  ==========================================新增模块==========================================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):#类的初始化函数，定义了该模块需要用到的参数和层
        super().__init__()#必须先调用父类的初始化函数
        self.weight = nn.Parameter(torch.ones(normalized_shape))#定义一个可学习的缩放参数 weight 初始值为1
        self.bias = nn.Parameter(torch.zeros(normalized_shape))#定义一个可学习的偏置参数bias,初始值为0
        self.eps = eps#防止分母出现零
        self.data_format = data_format#数据格式["channels_last", "channels_first"]
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError#格式不对就报错
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":# 如果数据格式是 (N, H, W, C)，直接使用PyTorch官方的layer_norm
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果数据格式是 (N, C, H, W)，这是PyTorch中CNN的常用格式，需要手动实现归一化
            u = x.mean(1, keepdim=True)# 沿着通道维度(C)计算均值
            s = (x - u).pow(2).mean(1, keepdim=True) # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps)# 归一化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]# 应用可学习的缩放和偏置
            return x
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim, out_dim, down_kernel, down_stride):
        super().__init__()

        self.lconv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=0, groups=in_dim)
        )
        self.hconv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=0, groups=in_dim)
        )#深度卷积改为上面哪个
        # self.lconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        # self.hconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim)
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(in_dim, in_dim, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2,
                              groups=in_dim)
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        B, C, H, W = x.shape

        dx = self.down(x)
        udx = F.interpolate(dx, size=(H, W), mode='bilinear', align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))

        return out
class AFE(nn.Module):
    def __init__(self, dim, kernel_size=3):#此处传入的也是3，默认也是为3

        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim,padding_mode='reflect')
        self.proj1 = nn.Conv2d(dim, dim // 2, 1, padding=0)
        self.proj2 = nn.Conv2d(dim, dim, 1, padding=0)

        self.ctx_conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=4,padding_mode='reflect')

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")

        self.enhance = FeatureRefinementModule(in_dim=dim // 2, out_dim=dim // 2, down_kernel=3, down_stride=2)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        y=x
        x = x + self.norm1(self.act(self.dwconv(x)))
        x = self.norm2(self.act(self.proj1(x)))
        ctx = self.norm3(self.act(self.ctx_conv(x)))  #SCM模块

        enh_x = self.enhance(x)                       #FRM模块,此处传过去的是dim/2 dim/2 down_kernel=3 down_down_stride=2
        x =y+ self.act(self.proj2(torch.cat([ctx, enh_x], dim=1)))
        return x
# ==========================================================================新增模块==================================