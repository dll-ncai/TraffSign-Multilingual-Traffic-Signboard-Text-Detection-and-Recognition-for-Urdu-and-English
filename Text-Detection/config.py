# data-config
import numpy as np

train_data_path = './dataset/train/'
train_batch_size_per_gpu = 8  # 14
num_workers = 2  # 24
gpu_ids = [0]  # [0,1,2,3]
gpu = 1  # 4
input_size = 256
background_ratio = 3. / 8
random_scale = np.array([0.5, 1, 2.0, 3.0])
geometry = 'RBOX'
max_image_large_side = 1280
max_text_size = 800
min_text_size = 10
min_crop_side_ratio = 0.1
means=[100, 100, 100]
pretrained = True
pretrained_basemodel_path = './tmp/epoch_3000_checkpoint.pth.tar'
pre_lr = 1e-4
lr = 1e-3
decay_steps = 50
decay_rate = 0.97
init_type = 'xavier'
resume = True
checkpoint = './tmp/epoch_3000_checkpoint.pth.tar'
max_epochs = 3000
l2_weight_decay = 1e-6
print_freq = 10
save_eval_iteration = 50
save_model_path = './tmp/'
test_img_path = './demo/test_img/'
res_img_path = './demo/result_img/'
write_images = True
score_map_thresh = 0.8
box_thresh = 0.1
nms_thres = 0.2
compute_hmean_path = './dataset/test_compute_hmean/'