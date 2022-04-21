from dataset.test_compute_hmean import rrc_evaluation_funcs, script
import config as cfg
import os
import glob
import zipfile
import time

gt_file_path = os.path.join(cfg.compute_hmean_path, 'gt.zip')
submit_file_path = os.path.join(cfg.compute_hmean_path, 'submit.zip')
log_file_path = os.path.join(cfg.compute_hmean_path, 'log_epoch_hmean.txt')
result_dir_path = cfg.compute_hmean_path

print('EAST <==> TEST <==> Compute Humean <==> Begin')

with zipfile.ZipFile(submit_file_path, 'w') as azip:
    txt_files = []
    txt_files.extend(glob.glob(
        os.path.join(cfg.res_img_path, '*.{}'.format('txt'))))
    for txt_name in txt_files:
        azip.write(txt_name, os.path.basename(txt_name), compress_type=zipfile.ZIP_LZMA)

resDict = rrc_evaluation_funcs.main_evaluation({'g': gt_file_path, 's': submit_file_path, 'o': result_dir_path},
                                               script.default_evaluation_params, script.validate_data,
                                               script.evaluate_method)

# print(resDict)
recall = resDict['method']['recall']
precision = resDict['method']['precision']
hmean = resDict['method']['hmean']

# print('EAST <==> Evaluation <==> Precision:%.4f Recall:%.4f Hmean %.4f <==> Done' % (precision, recall,
#                                                                                              hmean))
with open(log_file_path, 'a') as f:

    f.write(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    f.write('\nEAST <==> Evaluation <==> Precision:{:.4f} Recall:{:.4f} Hmean{:.4f} <==> Done\n'
            .format(precision, recall, hmean))

print('\nEAST <==> TEST <==> Compute Humean <==> End')
