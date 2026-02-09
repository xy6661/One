class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        # self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        # --- 开始修改 ---
        # 1. 保留Softmax，用于密集注意力分支
        self.sm = nn.Softmax(dim=-1)

        # 2. 新增ReLU激活，用于稀疏注意力分支
        self.relu = nn.ReLU()

        # 3. 新增一个可学习的参数`w`，包含两个值，分别对应两个分支的权重
        #    用nn.Parameter封装，模型在训练时会自动更新它
        self.w = nn.Parameter(torch.ones(2))
        # --- 修改结束 ---

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
        attn_sparse = self.relu(raw_attn) ** 2

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