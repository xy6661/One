python train.py \
--content_path /mnt/datasets/COCO/train2017 \
--style_path /mnt/datasets/wikistyle/cityscape \
--name AdaAttN_test \
--model adaattn \
--dataset_mode unaligned \
--no_dropout \
--load_size 512 \
--crop_size 256 \
--image_encoder_path ./models/vgg_normalised.pth \
--gpu_ids $1 \
--batch_size 4 \
--n_epochs 2 \
--n_epochs_decay 3 \
--display_freq 1 \
--display_port 8097 \
--display_env AdaAttN \
--lambda_local 3 \
--lambda_global 10 \
--lambda_content 0 \
--shallow_layer \
--skip_connection_3 \
--display_id -1


