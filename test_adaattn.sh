python test.py \
--content_path /mnt/harddisk2/Zhangmengge/codespace/test/test05/content/ \
--style_path /mnt/harddisk2/Zhangmengge/codespace/test/test05/style/  \
--name AdaAttN_test \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path ./models/vgg_normalised.pth \
--gpu_ids 0 \
--skip_connection_3 \
--shallow_layer


#在 test.py 脚本中，程序会首先解析命令行传入的参数。这些参数定义了测试的行为，其中与模型和权重加载最相关的参数在 options/base_options.py 中定义：
#--name: 实验的名称。这个名称非常重要，因为它决定了程序会去哪个文件夹下寻找要加载的模型权重。在 test_adaattn.sh 的例子中，名称是 AdaAttN。
#--checkpoints_dir: 存放所有模型和权重的根目录。默认值是 ./checkpoints。
#--model: 使用的模型的名称。在这里是 adaattn。
#--epoch: 加载哪个 epoch 的模型。默认是 latest，即加载最新的模型。
#--load_iter: 加载哪个迭代次数的模型。如果这个值大于 0，程序会加载 iter_[load_iter] 的模型。


#模型加载的核心逻辑位于 models/base_model.py 中的 setup 和 load_networks 方法。
#test.py 在初始化模型后会调用 model.setup(opt)。
#setup 方法会调用 self.load_networks(load_suffix)。
#load_networks 方法会遍历 self.model_names 列表，并根据 epoch 或 iteration 从磁盘加载每个网络的权重。权重的加载路径组合方式为 [checkpoints_dir]/[name]/[epoch]_net_[model_name].pth。


#对于 adaattn 模型，其具体的网络组件在 models/adaattn_model.py 中定义。
#在 AdaAttNModel 的 __init__ 方法中，self.model_names 被设置为 ['decoder', 'transformer']。
#如果命令行参数中包含了 --skip_connection_3，那么 'adaattn_3' 也会被添加到 self.model_names 中。
#另外，还有一个重要的部分是 图像编码器 (Image Encoder)，它是一个预训练的 VGG 网络。这个网络的权重是通过 --image_encoder_path 参数指定的路径加载的，加载代码也在 adaattn_model.py 中：