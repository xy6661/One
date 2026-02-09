def forward(self):
    self.c_feats = self.encode_with_intermediate(self.c)
    self.s_feats = self.encode_with_intermediate(self.s)
    s_att_3 = self.s_feats[2]
    s_att_4 = self.s_feats[3]
    s_att_5 = self.s_feats[4]#Vgg中得来的特征

    s_feats_att3 = self.net_style_attn_3(s_att_3)
    s_feats_att4 = self.net_style_attn_4(s_att_4)
    s_feats_att5 = self.net_style_attn_5(s_att_5)#这几个特征经过DilatedMDTA强化，未来的V
    # ////////////////////////////////////////////下面三行后加为了算id-loss作为输入两个都是内容图片的V
    c_feats_att3 = self.net_style_attn_3(self.c_feats[2])
    c_feats_att4 = self.net_style_attn_4(s_att_4)
    c_feats_att5 = self.net_style_attn_5(self.c_feats[4])

    c_feats_k2 = self.net_afe_k2(self.get_key(self.c_feats, 2, self.opt.shallow_layer))
    s_feats_k2 = self.net_afe_k2(self.get_key(self.s_feats, 2, self.opt.shallow_layer))
    c_feats_k3 = self.net_afe_k3(self.get_key(self.c_feats, 3, self.opt.shallow_layer))
    s_feats_k3 = self.net_afe_k3(self.get_key(self.s_feats, 3, self.opt.shallow_layer))
    c_feats_k4 = self.net_afe_k4(self.get_key(self.c_feats, 4, self.opt.shallow_layer))
    s_feats_k4 = self.net_afe_k4(self.get_key(self.s_feats, 4, self.opt.shallow_layer))
    # 后三层特征先拼接然后 通过AFE进行强化，作为骨干，产生softmax矩阵那两哥们

    c_feats_2 = self.net_afe_2(self.c_feats[2])
    c_feats_3 = self.net_afe_3(self.c_feats[3])
    c_feats_4 = self.net_afe_4(self.c_feats[4])#内容特征通过AFE强化了一次未来的主干信息
    # ////////////////////////////////////////////下面三行后加为了算id-loss作为输入两个都是风格图片的主干
    s_feats_2 = self.net_afe_2(s_att_3)
    s_feats_3 = self.net_afe_3(s_att_4)
    s_feats_4 = self.net_afe_4(s_att_5)



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

# =================以下为新添id===================================================
#     ====================cc================================
    if self.opt.skip_connection_3:
        # inplane=256，keyplane=256+128+64
        #这里面的特征原本是多层拼接的结构（1，2，3，4）分别为（内容->进行调制，风格v，多层内容q，多层风格q）
        #如今是（AFE内容->进行调制，DiaoGSA风格v，AFE多层内容q，AFE多层风格q）
        #但此处是算身份，稍稍优点不同
        cc_adain_feat_3 = self.net_adaattn_3(c_feats_2, c_feats_att3,
                                            c_feats_k2,
                                            c_feats_k2, self.seed)
    else:
        c_adain_feat_3 = None
    cc = self.net_transformer(c_feats_3, c_feats_att4, c_feats_4, c_feats_att5,
                              c_feats_k3,
                              c_feats_k3,
                              c_feats_k4,
                              c_feats_k4, self.seed)
    self.cc = self.net_decoder(cc, cc_adain_feat_3)
    # =====================ss=================================================
    if self.opt.skip_connection_3:
        # inplane=256，keyplane=256+128+64
        ss_adain_feat_3 = self.net_adaattn_3(s_feats_2, s_feats_att3,
                                            s_feats_k2,
                                            s_feats_k2, self.seed)
    else:
        c_adain_feat_3 = None
    ss = self.net_transformer(s_feats_3, s_feats_att4, s_feats_4, s_feats_att5,
                              s_feats_k3,
                              s_feats_k3,
                              s_feats_k4,
                              s_feats_k4, self.seed)
    self.cs = self.net_decoder(ss, c_adain_feat_3)
    