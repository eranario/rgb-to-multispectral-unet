import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class PotatoDataset(Dataset):

    def __init__(self, rgb_dir, spectral_dir, transform=None, mode='train', align=True):
        """Multispectral Potato Detection and Classification Dataset"""
        self.mode = mode
        self.transform = transform
        self.align = align
        self.channels = ['Green_Channel', 'Near_Infrared_Channel', 'Red_Channel', 'Red_Edge_Channel']

        # initialize file lists
        folder_set = 'Train_Images' if mode == 'train' else 'Test_Images'
        self.rgb_files = [f for f in os.listdir(os.path.join(rgb_dir, folder_set)) if f.endswith('.jpg')]
        self.spectral_files = {channel: [
            f for f in os.listdir(os.path.join(spectral_dir, channel, folder_set)) if f.endswith('.jpg')
        ] for channel in self.channels}

        # preload and align images using multiprocessing
        args_list = [
            (rgb_dir, spectral_dir, folder_set, rgb_name, idx, self.channels, self.spectral_files, self.align)
            for idx, rgb_name in enumerate(self.rgb_files)
        ]

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_image, args_list), desc="Loading data", total=len(self.rgb_files)))

        self.data = results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_image, spectral_images = self.data[idx]

        # convert to PIL
        rgb_image = Image.fromarray(rgb_image).convert('RGB')
        spectral_images = [Image.fromarray(img) for img in spectral_images]

        # apply transformations
        if self.transform:
            rgb_image = self.transform(rgb_image)
            spectral_images = [self.transform(img) for img in spectral_images]

        return (rgb_image, *spectral_images)

    @staticmethod
    def process_image(args):
        """
        Process a single image (alignment and resizing).
        """
        rgb_dir, spectral_dir, folder_set, rgb_name, idx, channels, spectral_files, align = args

        # read the RGB image
        rgb_path = os.path.join(rgb_dir, folder_set, rgb_name)
        rgb_im = cv2.imread(rgb_path)

        # resize RGB to match spectral dimensions
        height, width = cv2.imread(
            os.path.join(spectral_dir, channels[0], folder_set, spectral_files[channels[0]][idx]),
            cv2.IMREAD_GRAYSCALE
        ).shape
        rgb_resized = cv2.resize(rgb_im, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb_gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)

        # process spectral images
        spectral_images = []
        for channel in channels:
            spectral_path = os.path.join(spectral_dir, channel, folder_set, spectral_files[channel][idx])
            spectral_im = cv2.imread(spectral_path, cv2.IMREAD_GRAYSCALE)
            if align:
                aligned_image = PotatoDataset.align_images(rgb_gray, spectral_im)
                spectral_images.append(aligned_image)
            else:
                spectral_images.append(spectral_im)

        return (rgb_resized, spectral_images)

    @staticmethod
    def align_images(base_img, img_to_align):
        """
        Align img_to_align to base_img using ORB keypoints.
        """
        orb = cv2.ORB_create(6000)
        keypoints1, descriptors1 = orb.detectAndCompute(base_img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img_to_align, None)

        # match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # extract points
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # homography
        h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
        aligned_img = cv2.warpPerspective(img_to_align, h, (base_img.shape[1], base_img.shape[0]))
        return aligned_img
