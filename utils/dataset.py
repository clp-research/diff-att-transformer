# IMPORTS
import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import sys

sys.path.append('.')


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    IMPORTANT: Dataset offers the possibility to return the image pair either splitted (img_before, img_after) or stacked (img_stacked)
    """

    def __init__(self, data_folder, data_name, split, transform=None, img_return_mode='STACKED'):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        :param img_return_mode: return mode of image, one of 'SPLITTED' or 'STACKED', SPLITTED returns (img_before, img_after) or STACKED returns (img_stacked)
        """
        self.split_name = split
        assert self.split_name in {'TRAIN', 'VAL', 'TEST_SC', "TEST_NSC"}
        if self.split_name.startswith("TEST"):
            raise Exception("Please use TestDataset for the test split!")
        self.img_return_mode = img_return_mode
        assert self.img_return_mode in {'SPLITTED', 'STACKED'}

        # Open hdf5 files where images are stored, separate image_before and image_after
        self.h1 = h5py.File(os.path.join(data_folder, self.split_name + '_IMAGES_BEFORE_' + data_name + '.hdf5'), 'r')
        self.imgs_before = self.h1['images_before']
        self.h2 = h5py.File(os.path.join(data_folder, self.split_name + '_IMAGES_AFTER_' + data_name + '.hdf5'), 'r')
        self.imgs_after = self.h2['images_after']

        self.captions_per_image = self.h1.attrs['captions_per_image_pair']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split_name + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split_name + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        if self.split_name in ["TRAIN", "VAL"]:
            # We want to train on all captions
            self.dataset_size = len(self.captions)
        else:
            # We want to evaluate only once per image-pair
            self.dataset_size = len(self.imgs_before)

    def __getitem__(self, i):
        if self.img_return_mode is 'SPLITTED':
            if self.split_name in ["TRAIN", "VAL"]:
                # For training we "iterate over the captions" and thus use the same image "cpi"-times
                image_index = i // self.captions_per_image
            else:
                # For evaluation, we simply use each image once
                image_index = i

            img_before = torch.FloatTensor(self.imgs_before[image_index] / 255.)
            img_after = torch.FloatTensor(self.imgs_after[image_index] / 255.)
            if self.transform is not None:
                img_before = self.transform(img_before)
                img_after = self.transform(img_after)

            # For validation and testing, also return all 'captions_per_image_pair' captions to find BLEU-4 score
            # This might be worse than the DUDA evaluation, because we fix to a maximum number of captions
            # for example in clevr_change there are sometimes up to 9 captions, but we only evaluate for 5 here.
            caption_from_idx = image_index * self.captions_per_image
            caption_to_idx = caption_from_idx + self.captions_per_image
            img_references = self.captions[caption_from_idx:caption_to_idx]
            img_references = torch.LongTensor(img_references)

            if self.split_name in ["TRAIN", "VAL"]:
                # For train and validate we iterate over the captions
                caption = torch.LongTensor(self.captions[i])
                caplen = torch.LongTensor([self.caplens[i]])
                return img_before, img_after, caption, caplen, img_references
            else:
                # For test we iterate over the image-pair, so that there is no single caption to be returned
                return img_before, img_after, img_references

        else:
            img_before = torch.FloatTensor(self.imgs_before[i // self.captions_per_image] / 255.)
            img_after = torch.FloatTensor(self.imgs_after[i // self.captions_per_image] / 255.)
            img_stacked = torch.cat((img_before, img_after), 2)  # concatenate along the horizontal axis
            assert img_stacked.shape == (3, 360, 960)  #
            if self.transform is not None:
                img_stacked = self.transform(img_stacked)

            caption = torch.LongTensor(self.captions[i])

            caplen = torch.LongTensor([self.caplens[i]])

            if self.split_name is 'TRAIN':
                return img_stacked, caption, caplen
            else:
                caption_from_idx = ((i // self.captions_per_image) * self.captions_per_image)
                caption_to_idx = (caption_from_idx + self.captions_per_image)
                # For validation of testing, also return all 'captions_per_image_pair' captions to find BLEU-4 score
                all_captions = torch.LongTensor(self.captions[caption_from_idx:caption_to_idx])
                return img_stacked, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class TestDataset(Dataset):
    """
    """

    def __init__(self, data_folder, data_name, split_name, transform=None):
        self.split_name = split_name
        assert self.split_name in {'TRAIN', 'VAL', 'TEST_SC', "TEST_NSC"}
        if not self.split_name.startswith("TEST"):
            raise Exception("Please use this dataset only with the test split")

        self.file = h5py.File(os.path.join(data_folder, self.split_name + '_STORE_' + data_name + '.hdf5'), 'r')
        self.image_pairs = self.file["image_pairs"]
        with open(os.path.join(data_folder, self.split_name + '_STORE_' + data_name + '.json')) as f:
            self.captions = json.load(f)
        self.transform = transform
        self.dataset_size = len(self.image_pairs)

    def get_image_pair(self, idx, image_length=3 * 320 * 480, image_shape=(3, 320, 480)):
        image_pair = self.image_pairs[idx]
        img_before = image_pair[:image_length].reshape(image_shape)
        img_after = image_pair[image_length:].reshape(image_shape)
        img_before = torch.FloatTensor(img_before / 255.)
        img_after = torch.FloatTensor(img_after / 255.)
        return img_before, img_after

    def __getitem__(self, idx):
        img_before, img_after = self.get_image_pair(idx)
        if self.transform is not None:
            img_before = self.transform(img_before)
            img_after = self.transform(img_after)
        img_references = self.captions[idx]
        img_references = torch.LongTensor(img_references)
        return img_before, img_after, img_references

    def __len__(self):
        return self.dataset_size
