import torch
from PIL import Image
import numpy as np
import os
from glob import glob

class MakeImage():
    """
    this class used to make list of data for ImageNet
    """
    def __init__(self, dataset_dir, split=0.8):
        self.train_data = []
        self.val_data = []
        self.category = ["none"] * 256

        self.get_data_sequence(dataset_dir, split)

    def get_data_sequence(self, root, split):
        class_dirs = os.listdir(root)
        
        for class_dir in class_dirs:
            class_id = int(class_dir.split(".")[0]) - 1
            class_cat = class_dir.split(".")[1]
            
            

            if class_id != 256:
                self.get_img_paths(os.path.join(root, class_dir), class_id, split)
                self.category[class_id] =  class_cat
            

    def get_img_paths(self, class_dir, class_id, split):
        files_path = glob(class_dir + "/*.jpg")
        class_train_num = int(len(files_path)*split)

        class_train_data =[[i, class_id] for i in files_path[:class_train_num]]
        class_val_data = [[i, class_id] for i in files_path[class_train_num:]]

        self.train_data += class_train_data
        self.val_data += class_val_data

    def get_data(self):

        return self.train_data, self.val_data, self.category


class Caltech(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, phase, transform=None):
        self.train, self.val, self.category = MakeImage(dataset_dir).get_data()
        self.all_data = {"train": self.train, "val": self.val}[phase]
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item_id):
        image_root = self.all_data[item_id][0]
        image = Image.open(image_root).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.all_data[item_id][1]
        label = torch.from_numpy(np.array(label))
        return image, label


if __name__ == "__main__":
    dataset = Caltech("/home/coder/projects/data/256_ObjectCategories", "val")