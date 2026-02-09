# """General-purpose test script for image-to-image translation.
#
# Once you have trained your model with train.py, you can use this script to test the model.
# It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
#
# It first creates model and dataset given the option. It will hard-code some parameters.
# It then runs inference for '--num_test' images and save results to an HTML file.
#
# Example (You need to train models first or download pre-trained models from our website):
#     Test a CycleGAN model (both sides):
#         python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
#
#     Test a CycleGAN model (one side only):
#         python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
#
#     The option '--model test' is used for generating CycleGAN results only for one side.
#     This option will automatically set '--dataset_mode single', which only loads the images from one set.
#     On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
#     which is sometimes unnecessary. The results will be saved at ./results/.
#     Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
#
#     Test a pix2pix model:
#         python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
#
# See options/base_options.py and options/test_options.py for more test options.
# See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
# See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
# """
# # import os
# # from options.test_options import TestOptions
# # from data import create_dataset
# # from models import create_model
# # from util.visualizer import save_images
# # from util import html
# #
# #
# # if __name__ == '__main__':
# #     opt = TestOptions().parse()  # get test options
# #     # hard-code some parameters for test
# #     opt.num_threads = 0   # test code only supports num_threads = 0
# #     opt.batch_size = 1    # test code only supports batch_size = 1
# #     opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
# #     opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
# #     opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
# #     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
# #     model = create_model(opt)      # create a model given opt.model and other options
# #     model.setup(opt)               # regular setup: load and print networks; create schedulers
# #     # create a website
# #     web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
# #     if opt.load_iter > 0:  # load_iter is 0 by default
# #         web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
# #     print('creating web directory', web_dir)
# #     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
# #     # test with eval mode. This only affects layers like batchnorm and dropout.
# #     # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
# #     # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
# #     if opt.eval:
# #         model.eval()
# #     for i, data in enumerate(dataset):
# #         if i >= opt.num_test:  # only apply our model to opt.num_test images.
# #             break
# #         model.set_input(data)  # unpack data from data loader
# #         model.test()           # run inference
# #         visuals = model.get_current_visuals()  # get image results
# #         img_path = model.get_image_paths()     # get image paths
# #         if i % 5 == 0:  # save images to an HTML file
# #             print('processing (%04d)-th image... %s' % (i, img_path))
# #         save_images(webpage, visuals, img_path, width=opt.display_winsize)
# #     webpage.save()  # save the HTML
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
import torch


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # 新增：用于累加/计数（可用于多张图像场景）
    total_time = 0.0
    count = 0
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        model.test()           # run inference

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        total_time += elapsed
        count += 1
        print("Inference time: %.4f seconds" % elapsed)

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    if count > 0:
        print("Average inference time: %.4f seconds (over %d runs)" % (total_time / count, count))
    webpage.save()  # save the HTML

# import os
# import time  # 1. 导入 time 模块
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from util import html
# from util.util import calculate_ssim  # 2. 导入我们新增的 SSIM 函数
# import numpy as np # 导入 numpy 用于数学计算
#
# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     # hard-code some parameters for test
#     opt.num_threads = 0
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.no_flip = True
#     opt.display_id = -1
#     dataset = create_dataset(opt)
#     model = create_model(opt)
#     model.setup(opt)
#
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
#     if opt.load_iter > 0:
#         web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
#     print('creating web directory', web_dir)
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#
#     if opt.eval:
#         model.eval()
#
#     # --- 新增的统计变量 ---
#     # 用于计算最终总平均值的累加器
#     total_content_loss = 0.0
#     total_style_loss = 0.0
#     total_ssim_score = 0.0
#     total_image_count = 0
#
#     # 用于计算每张内容图平均值的累加器
#     per_content_content_loss = 0.0
#     per_content_style_loss = 0.0
#     per_content_ssim_score = 0.0
#     per_content_image_count = 0
#
#     last_content_path = ""  # 用于判断内容图是否发生变化
#
#     # --- 主循环 ---
#     for i, data in enumerate(dataset):
#         if i >= opt.num_test:
#             break
#
#         model.set_input(data)
#
#         # 判断内容图是否已经更换
#         current_content_path = model.get_image_paths()[0]  # 内容图路径
#         if last_content_path != "" and current_content_path != last_content_path:
#             # 如果更换了，就打印上一张内容图的平均统计数据
#             avg_c_loss = per_content_content_loss / per_content_image_count
#             avg_s_loss = per_content_style_loss / per_content_image_count
#             avg_ssim = per_content_ssim_score / per_content_image_count
#             print("\n" + "=" * 20 + " Average for Content Image " + "=" * 20)
#             print("Content Image: %s" % os.path.basename(last_content_path))
#             print("  > Avg Content Loss: %.4f" % avg_c_loss)
#             print("  > Avg Style Loss:   %.4f" % avg_s_loss)
#             print("  > Avg SSIM:         %.4f" % avg_ssim)
#             print("=" * 65 + "\n")
#
#             # 重置"每张内容图"的累加器
#             per_content_content_loss = 0.0
#             per_content_style_loss = 0.0
#             per_content_ssim_score = 0.0
#             per_content_image_count = 0
#
#         # a. 计时并运行推理
#         start_time = time.time()
#         model.test()
#         processing_time = end_time = time.time() - start_time
#
#         # b. 获取特征并计算损失
#         stylized_feats = model.encode_with_intermediate(model.cs)
#         model.content_loss(stylized_feats)
#         model.style_loss(stylized_feats)
#
#         # c. 提取各项指标值
#         content_loss_val = model.loss_content.item()
#         style_loss_val = model.loss_global.item()
#         ssim_score_val = calculate_ssim(model.c, model.cs).item()
#
#         # d. 累加到统计变量中
#         total_content_loss += content_loss_val
#         total_style_loss += style_loss_val
#         total_ssim_score += ssim_score_val
#         total_image_count += 1
#         per_content_content_loss += content_loss_val
#         per_content_style_loss += style_loss_val
#         per_content_ssim_score += ssim_score_val
#         per_content_image_count += 1
#
#         visuals = model.get_current_visuals()
#         img_paths = model.get_image_paths()
#
#         # e. 打印当前这次风格化的详细信息
#         content_name = os.path.basename(img_paths[0])
#         style_name = os.path.basename(data['s_path'][0])  # 从数据中获取风格图路径
#         print('(%04d/%d) Content: %s | Style: %s' % (i + 1, len(dataset), content_name, style_name))
#         print('  > Content Loss: %.4f | Style Loss: %.4f | SSIM: %.4f' % (
#         content_loss_val, style_loss_val, ssim_score_val))
#
#         # 保存图片
#         save_images(webpage, visuals, img_paths, width=opt.display_winsize)
#
#         last_content_path = current_content_path
#
#     # --- 循环结束后 ---
#     # 打印最后一张内容图的平均统计数据
#     if per_content_image_count > 0:
#         avg_c_loss = per_content_content_loss / per_content_image_count
#         avg_s_loss = per_content_style_loss / per_content_image_count
#         avg_ssim = per_content_ssim_score / per_content_image_count
#         print("\n" + "=" * 20 + " Average for Content Image " + "=" * 20)
#         print("Content Image: %s" % os.path.basename(last_content_path))
#         print("  > Avg Content Loss: %.4f" % avg_c_loss)
#         print("  > Avg Style Loss:   %.4f" % avg_s_loss)
#         print("  > Avg SSIM:         %.4f" % avg_ssim)
#         print("=" * 65 + "\n")
#
#     # 打印所有测试图片的最终总平均值
#     if total_image_count > 0:
#         final_avg_c_loss = total_content_loss / total_image_count
#         final_avg_s_loss = total_style_loss / total_image_count
#         final_avg_ssim = total_ssim_score / total_image_count
#         print("\n" + "#" * 25 + " Final Overall Average " + "#" * 25)
#         print("Total Images Processed: %d" % total_image_count)
#         print("  > Overall Avg Content Loss: %.4f" % final_avg_c_loss)
#         print("  > Overall Avg Style Loss:   %.4f" % final_avg_s_loss)
#         print("  > Overall Avg SSIM:         %.4f" % final_avg_ssim)
#         print("#" * 75 + "\n")
#
#     webpage.save()  # save the HTML

# import os
# import time
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from util import html
# from util.util import calculate_ssim  # 导入我们新增的 SSIM 函数
# import numpy as np  # 导入 numpy 用于数学计算
#
# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     # hard-code some parameters for test
#     opt.num_threads = 0
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.no_flip = True
#     opt.display_id = -1
#     dataset = create_dataset(opt)
#     model = create_model(opt)
#     model.setup(opt)
#
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
#     if opt.load_iter > 0:
#         web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
#     print('creating web directory', web_dir)
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#
#     if opt.eval:
#         model.eval()
#
#     # --- 新增的统计变量 ---
#     total_content_loss, total_style_loss, total_ssim_score, total_time = 0.0, 0.0, 0.0, 0.0
#     total_image_count = 0
#     per_content_content_loss, per_content_style_loss, per_content_ssim_score, per_content_time = 0.0, 0.0, 0.0, 0.0
#     per_content_image_count = 0
#     last_content_path = ""
#
#     # --- 主循环 ---
#     for i, data in enumerate(dataset):
#         if i >= opt.num_test:
#             break
#
#         model.set_input(data)
#
#         # 判断内容图是否已经更换
#         current_content_path = data['c_path'][0]  # 从数据中获取内容图路径
#         if last_content_path != "" and current_content_path != last_content_path:
#             # 打印上一张内容图的平均统计数据
#             avg_c_loss = per_content_content_loss / per_content_image_count
#             avg_s_loss = per_content_style_loss / per_content_image_count
#             avg_ssim = per_content_ssim_score / per_content_image_count
#             avg_time = per_content_time / per_content_image_count
#             print("\n" + "=" * 20 + " Average for Content Image " + "=" * 20)
#             print("Content Image: %s" % os.path.basename(last_content_path))
#             print("  > Avg Time/Img:     %.4f s" % avg_time)
#             print("  > Avg Content Loss: %.4f" % avg_c_loss)
#             print("  > Avg Style Loss:   %.4f" % avg_s_loss)
#             print("  > Avg SSIM:         %.4f" % avg_ssim)
#             print("=" * 65 + "\n")
#
#             # 重置"每张内容图"的累加器
#             per_content_content_loss, per_content_style_loss, per_content_ssim_score, per_content_time = 0.0, 0.0, 0.0, 0.0
#             per_content_image_count = 0
#
#         # a. 计时并运行推理
#         start_time = time.time()
#         model.test()
#         end_time = time.time()
#         processing_time = end_time - start_time  # 计算单张图片处理时间
#
#         # b. 获取特征并计算损失
#         stylized_feats = model.encode_with_intermediate(model.cs)
#         model.content_loss(stylized_feats)
#         model.style_loss(stylized_feats)
#
#         # c. 提取各项指标值
#         content_loss_val = model.loss_content.item()
#         style_loss_val = model.loss_global.item()
#         ssim_score_val = calculate_ssim(model.c, model.cs).item()
#         # ssim_score_val = calculate_ssim(model.c, model.cs)
#         # d. 累加到统计变量中
#         total_content_loss += content_loss_val
#         total_style_loss += style_loss_val
#         total_ssim_score += ssim_score_val
#         total_time += processing_time
#         total_image_count += 1
#         per_content_content_loss += content_loss_val
#         per_content_style_loss += style_loss_val
#         per_content_ssim_score += ssim_score_val
#         per_content_time += processing_time
#         per_content_image_count += 1
#
#         visuals = model.get_current_visuals()
#
#         # e. 打印当前这次风格化的详细信息
#         content_name = os.path.basename(current_content_path)
#         style_name = os.path.basename(data['s_path'][0])
#         print('(%04d/%d) Content: %s | Style: %s' % (i + 1, len(dataset), content_name, style_name))
#         print('  > Time: %.4f s | Content Loss: %.4f | Style Loss: %.4f | SSIM: %.4f' % (
#         processing_time, content_loss_val, style_loss_val, ssim_score_val))
#
#         # 保存图片
#         # save_images(webpage, visuals, data['c_path'], width=opt.display_winsize)
#         # ...
#         # 保存图片
#         # 我们使用 model.get_image_paths() 来获取由数据集加载器生成的唯一文件名
#         save_images(webpage, visuals, model.get_image_paths(), width=opt.display_winsize)
#         last_content_path = current_content_path
#
#     # --- 循环结束后 ---
#     # 打印最后一张内容图的平均统计数据
#     if per_content_image_count > 0:
#         avg_c_loss = per_content_content_loss / per_content_image_count
#         avg_s_loss = per_content_style_loss / per_content_image_count
#         avg_ssim = per_content_ssim_score / per_content_image_count
#         avg_time = per_content_time / per_content_image_count
#         print("\n" + "=" * 20 + " Average for Content Image " + "=" * 20)
#         print("Content Image: %s" % os.path.basename(last_content_path))
#         print("  > Avg Time/Img:     %.4f s" % avg_time)
#         print("  > Avg Content Loss: %.4f" % avg_c_loss)
#         print("  > Avg Style Loss:   %.4f" % avg_s_loss)
#         print("  > Avg SSIM:         %.4f" % avg_ssim)
#         print("=" * 65 + "\n")
#
#     # 打印所有测试图片的最终总平均值
#     if total_image_count > 0:
#         final_avg_c_loss = total_content_loss / total_image_count
#         final_avg_s_loss = total_style_loss / total_image_count
#         final_avg_ssim = total_ssim_score / total_image_count
#         final_avg_time = total_time / total_image_count
#         print("\n" + "#" * 25 + " Final Overall Average " + "#" * 25)
#         print("Total Images Processed: %d" % total_image_count)
#         print("  > Overall Avg Time/Img:     %.4f s" % final_avg_time)
#         print("  > Overall Avg Content Loss: %.4f" % final_avg_c_loss)
#         print("  > Overall Avg Style Loss:   %.4f" % final_avg_s_loss)
#         print("  > Overall Avg SSIM:         %.4f" % final_avg_ssim)
#         print("#" * 75 + "\n")
#
#     webpage.save()  # save the HTML

# import os
# import time
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images
# from util import html
# from util.util import calculate_ssim  # 导入我们新增的 SSIM 函数
# import numpy as np  # 导入 numpy 用于数学计算
#
# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     # hard-code some parameters for test
#     opt.num_threads = 0
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.no_flip = True
#     opt.display_id = -1
#     dataset = create_dataset(opt)
#     model = create_model(opt)
#     model.setup(opt)
#
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
#     if opt.load_iter > 0:
#         web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
#     print('creating web directory', web_dir)
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
#
#     if opt.eval:
#         model.eval()
#
#     # --- 新增的统计变量 ---
#     total_content_loss, total_style_loss, total_ssim_score, total_time = 0.0, 0.0, 0.0, 0.0
#     total_image_count = 0
#     per_content_content_loss, per_content_style_loss, per_content_ssim_score, per_content_time = 0.0, 0.0, 0.0, 0.0
#     per_content_image_count = 0
#     last_content_path = ""
#
#     # --- 主循环 ---
#     for i, data in enumerate(dataset):
#         if i >= opt.num_test:
#             break
#
#         model.set_input(data)
#
#         # 判断内容图是否已经更换
#         current_content_path = data['c_path'][0]  # 从数据中获取内容图路径
#         if last_content_path != "" and current_content_path != last_content_path:
#             # 打印上一张内容图的平均统计数据
#             avg_c_loss = per_content_content_loss / per_content_image_count
#             avg_s_loss = per_content_style_loss / per_content_image_count
#             avg_ssim = per_content_ssim_score / per_content_image_count
#             avg_time = per_content_time / per_content_image_count
#             print("\n" + "=" * 20 + " Average for Content Image " + "=" * 20)
#             print("Content Image: %s" % os.path.basename(last_content_path))
#             print("  > Avg Time/Img:     %.4f s" % avg_time)
#             print("  > Avg Content Loss: %.4f" % avg_c_loss)
#             print("  > Avg Style Loss:   %.4f" % avg_s_loss)
#             print("  > Avg SSIM:         %.4f" % avg_ssim)
#             print("=" * 65 + "\n")
#
#             # 重置"每张内容图"的累加器
#             per_content_content_loss, per_content_style_loss, per_content_ssim_score, per_content_time = 0.0, 0.0, 0.0, 0.0
#             per_content_image_count = 0
#
#         # a. 计时并运行推理
#         start_time = time.time()
#         model.test()
#         end_time = time.time()
#         processing_time = end_time - start_time  # 计算单张图片处理时间
#
#         # b. 获取特征并计算损失
#         stylized_feats = model.encode_with_intermediate(model.cs)
#         model.content_loss(stylized_feats)
#         model.style_loss(stylized_feats)
#
#         # c. 提取各项指标值
#         content_loss_val = model.loss_content.item()
#         style_loss_val = model.loss_global.item()
#         # ssim_score_val = calculate_ssim(model.c, model.cs).item()
#         ssim_score_val = calculate_ssim(model.c, model.cs)
#
#         # d. 累加到统计变量中
#         total_content_loss += content_loss_val
#         total_style_loss += style_loss_val
#         total_ssim_score += ssim_score_val
#         total_time += processing_time
#         total_image_count += 1
#         per_content_content_loss += content_loss_val
#         per_content_style_loss += style_loss_val
#         per_content_ssim_score += ssim_score_val
#         per_content_time += processing_time
#         per_content_image_count += 1
#
#         visuals = model.get_current_visuals()
#
#         # e. 打印当前这次风格化的详细信息
#         content_name = os.path.basename(current_content_path)
#         style_name = os.path.basename(data['s_path'][0])
#         print('(%04d/%d) Content: %s | Style: %s' % (i + 1, len(dataset), content_name, style_name))
#         print('  > Time: %.4f s | Content Loss: %.4f | Style Loss: %.4f | SSIM: %.4f' % (
#         processing_time, content_loss_val, style_loss_val, ssim_score_val))
#
#         # 保存图片
#         save_images(webpage, visuals, data['c_path'], width=opt.display_winsize)
#
#         last_content_path = current_content_path
#
#     # --- 循环结束后 ---
#     # 打印最后一张内容图的平均统计数据
#     if per_content_image_count > 0:
#         avg_c_loss = per_content_content_loss / per_content_image_count
#         avg_s_loss = per_content_style_loss / per_content_image_count
#         avg_ssim = per_content_ssim_score / per_content_image_count
#         avg_time = per_content_time / per_content_image_count
#         print("\n" + "=" * 20 + " Average for Content Image " + "=" * 20)
#         print("Content Image: %s" % os.path.basename(last_content_path))
#         print("  > Avg Time/Img:     %.4f s" % avg_time)
#         print("  > Avg Content Loss: %.4f" % avg_c_loss)
#         print("  > Avg Style Loss:   %.4f" % avg_s_loss)
#         print("  > Avg SSIM:         %.4f" % avg_ssim)
#         print("=" * 65 + "\n")
#
#     # 打印所有测试图片的最终总平均值
#     if total_image_count > 0:
#         final_avg_c_loss = total_content_loss / total_image_count
#         final_avg_s_loss = total_style_loss / total_image_count
#         final_avg_ssim = total_ssim_score / total_image_count
#         final_avg_time = total_time / total_image_count
#         print("\n" + "#" * 25 + " Final Overall Average " + "#" * 25)
#         print("Total Images Processed: %d" % total_image_count)
#         print("  > Overall Avg Time/Img:     %.4f s" % final_avg_time)
#         print("  > Overall Avg Content Loss: %.4f" % final_avg_c_loss)
#         print("  > Overall Avg Style Loss:   %.4f" % final_avg_s_loss)
#         print("  > Overall Avg SSIM:         %.4f" % final_avg_ssim)
#         print("#" * 75 + "\n")
#
#     webpage.save()  # save the HTML