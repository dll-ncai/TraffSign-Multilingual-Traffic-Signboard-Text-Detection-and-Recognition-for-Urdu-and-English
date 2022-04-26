from torch.utils import data
import numpy as np
import preprossing
import config as cfg


class custom_dset(data.Dataset):
    def __init__(self, transform=None):


        self.img_path_list = preprossing.get_images(cfg.train_data_path)
        self.transform = transform

    def __getitem__(self, index):

        status = True
        while status:
            img, img_path, score_map, geo_map, training_mask = preprossing.generator(
                index=index,
                input_size=cfg.input_size,
                background_ratio=cfg.background_ratio,
                random_scale=cfg.random_scale,
                image_list=self.img_path_list)

            if not (img is None):

                status = False
                if self.transform is not None:
                    img = self.transform(img)
                    score_map = self.transform(score_map)
                    geo_map = self.transform(geo_map)
                    training_mask = self.transform(training_mask)
                    return img, img_path, score_map, geo_map, training_mask

            else:
                index = np.random.randint(0, self.__len__())
                # print('Exception in getitem, and choose another index:{}'.format(index))

    def __len__(self):
        return len(self.img_path_list)

# img = bs * 512 * 512 *3
# score_map = bs* 128 * 128 * 1
# geo_map = bs * 128 * 128 * 5
# training_mask = bs * 128 * 128 * 1