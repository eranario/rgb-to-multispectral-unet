import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

class PotatoDataset(Dataset):
    
    def __init__(
        self, rgb_dir, spectral_dir, transform=None, mode='train', align=False, crop_factor=0.8, split_ratio=0.8, random_seed=42
    ):
        """Multispectral Potato Detection and Classification Dataset"""
        self.mode = mode
        self.transform = transform
        self.align = align
        self.crop_factor = crop_factor
        self.channels = ['Green_Channel', 'Near_Infrared_Channel', 'Red_Channel', 'Red_Edge_Channel']

        if mode == 'test':
            # Load test data from a separate folder
            self.rgb_files = sorted([f for f in os.listdir(os.path.join(rgb_dir, 'Test_Images')) if f.endswith('.jpg')])
            self.spectral_files = {channel: sorted([
                f for f in os.listdir(os.path.join(spectral_dir, channel, 'Test_Images')) if f.endswith('.jpg')
            ]) for channel in self.channels}
            folder_set = 'Test_Images'
        else:
            # Load training data
            all_rgb_files = sorted([f for f in os.listdir(os.path.join(rgb_dir, 'Train_Images')) if f.endswith('.jpg')])
            spectral_files = {channel: sorted([
                f for f in os.listdir(os.path.join(spectral_dir, channel, 'Train_Images')) if f.endswith('.jpg')
            ]) for channel in self.channels}

            # Split RGB files into train and val sets
            train_files, val_files = train_test_split(all_rgb_files, test_size=1 - split_ratio, random_state=random_seed)

            # Sort train and validation files after the split
            train_files = sorted(train_files)
            val_files = sorted(val_files)

            # Filter spectral files to match RGB splits
            train_spectral_files = {
                channel: [f for f in spectral_files[channel] if os.path.splitext(f)[0] in {os.path.splitext(r)[0] for r in train_files}]
                for channel in self.channels
            }
            val_spectral_files = {
                channel: [f for f in spectral_files[channel] if os.path.splitext(f)[0] in {os.path.splitext(r)[0] for r in val_files}]
                for channel in self.channels
            }

            # Assign files based on mode
            if mode == 'train':
                self.rgb_files = train_files
                self.spectral_files = train_spectral_files
            elif mode == 'val':
                self.rgb_files = val_files
                self.spectral_files = val_spectral_files
            else:
                raise ValueError("Mode must be 'train', 'val', or 'test'")

            folder_set = 'Train_Images'

        # Preload and align images using multiprocessing
        args_list = [
            (rgb_dir, spectral_dir, folder_set, rgb_name, idx, self.channels, self.spectral_files, self.align)
            for idx, rgb_name in enumerate(self.rgb_files)
        ]

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_image, args_list), desc=f"Loading {mode} data", total=len(self.rgb_files)))

        self.data = results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_image, spectral_images = self.data[idx]

        # Convert to PIL
        rgb_image = Image.fromarray(rgb_image).convert('RGB')
        spectral_images = [Image.fromarray(img) for img in spectral_images]

        # Apply transformations
        if self.transform:
            rgb_image = self.transform(rgb_image)
            spectral_images = [self.transform(img) for img in spectral_images]

        return (rgb_image, *spectral_images)

    def process_image(self, args):
        rgb_dir, spectral_dir, folder_set, rgb_name, idx, channels, spectral_files, align = args

        # Read the RGB image
        rgb_path = os.path.join(rgb_dir, folder_set, rgb_name)
        rgb_im = cv2.imread(rgb_path)

        # Ensure the base names match
        for channel in channels:
            spectral_file = spectral_files[channel][idx]
            if os.path.splitext(rgb_name)[0] != os.path.splitext(spectral_file)[0]:
                raise ValueError(
                    f"Mismatch detected: RGB file '{rgb_name}' does not match spectral file '{spectral_file}' in channel '{channel}'"
                )

        # Resize RGB to match spectral dimensions
        height, width = cv2.imread(
            os.path.join(spectral_dir, channels[0], folder_set, spectral_files[channels[0]][idx]),
            cv2.IMREAD_GRAYSCALE
        ).shape
        rgb_resized = cv2.resize(rgb_im, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb_gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)

        # Process spectral images
        spectral_images = []
        for channel in channels:
            spectral_path = os.path.join(spectral_dir, channel, folder_set, spectral_files[channel][idx])
            spectral_im = cv2.imread(spectral_path, cv2.IMREAD_GRAYSCALE)
            if align:
                aligned_image = PotatoDataset.align_images(rgb_gray, spectral_im)
                spectral_images.append(aligned_image)
            else:
                spectral_images.append(spectral_im)

        # Validate size consistency
        for channel, spectral_im in zip(channels, spectral_images):
            assert spectral_im.shape == rgb_resized.shape[:2], \
                f"Size mismatch: RGB {rgb_resized.shape[:2]} vs {channel} {spectral_im.shape}"

        # Crop images to the center
        crop_height = int(height * self.crop_factor)
        crop_width = int(width * self.crop_factor)
        crop_size = (crop_height, crop_width)
        rgb_cropped = PotatoDataset.center_crop(rgb_resized, crop_size)
        spectral_cropped = [PotatoDataset.center_crop(img, crop_size) for img in spectral_images]

        return (rgb_cropped, spectral_cropped)

    @staticmethod
    def align_images(base_img, img_to_align):
        """
        Align two images using SIFT keypoints and RANSAC.
        
        Args:
            base_img (numpy.ndarray): The base image to align to.
            img_to_align (numpy.ndarray): The image to align.
            
        Returns:
            numpy.ndarray: The aligned image.
        """ 
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_to_align, None)
        kp2, des2 = sift.detectAndCompute(base_img, None)

        matcher = cv2.FlannBasedMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # Filter good matches
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                    good.append(m)

        # Estimate homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp image
        aligned_img = cv2.warpPerspective(img_to_align, H, (base_img.shape[1], base_img.shape[0]))
        return aligned_img
    
    @staticmethod
    def center_crop(image, crop_size):
        """
        Crop the center of an image to the given size.
        
        Args:
            image (numpy.ndarray): The input image (H x W or H x W x C).
            crop_size (tuple): The desired output size (height, width).
        
        Returns:
            numpy.ndarray: The center-cropped image.
        """
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h + crop_h, start_w:start_w + crop_w]

class PotatoDatasetSpectra(Dataset):
    
    def __init__(
        self, rgb_dir, spectral_dir, spectral_file, transform=None, mode='train', align=False, crop_factor=0.8, split_ratio=0.8, random_seed=42, hsv_bounds=((35, 40, 40), (85, 255, 255))
    ):
        """Multispectral Potato Detection and Classification Dataset"""
        self.mode = mode
        self.transform = transform
        self.align = align
        self.crop_factor = crop_factor
        self.spectral_file = spectral_file
        self.channels = ['Green_Channel', 'Near_Infrared_Channel', 'Red_Channel', 'Red_Edge_Channel']
        self.hsv_lower, self.hsv_upper = hsv_bounds
        
        # load spectral data
        df_spectra = pd.read_csv(spectral_file, index_col=0)
        df_spectra_pruned = df_spectra[df_spectra.columns[2:]].reset_index(drop=True)
        spectra_array = df_spectra_pruned.values
        
        # get number of bands
        self.num_bands = spectra_array.shape[1]

        if mode == 'test':
            # load test data from a separate folder
            self.rgb_files = sorted([f for f in os.listdir(os.path.join(rgb_dir, 'Test_Images')) if f.endswith('.jpg')])
            self.spectral_files = {channel: sorted([
                f for f in os.listdir(os.path.join(spectral_dir, channel, 'Test_Images')) if f.endswith('.jpg')
            ]) for channel in self.channels}
            folder_set = 'Test_Images'
        else:
            # load training data
            all_rgb_files = sorted([f for f in os.listdir(os.path.join(rgb_dir, 'Train_Images')) if f.endswith('.jpg')])
            spectral_files = {channel: sorted([
                f for f in os.listdir(os.path.join(spectral_dir, channel, 'Train_Images')) if f.endswith('.jpg')
            ]) for channel in self.channels}

            # split RGB files into train and val sets
            train_files, val_files = train_test_split(all_rgb_files, test_size=1 - split_ratio, random_state=random_seed)

            # sort train and validation files after the split
            train_files = sorted(train_files)
            val_files = sorted(val_files)

            # filter spectral files to match RGB splits
            train_spectral_files = {
                channel: [f for f in spectral_files[channel] if os.path.splitext(f)[0] in {os.path.splitext(r)[0] for r in train_files}]
                for channel in self.channels
            }
            val_spectral_files = {
                channel: [f for f in spectral_files[channel] if os.path.splitext(f)[0] in {os.path.splitext(r)[0] for r in val_files}]
                for channel in self.channels
            }

            # assign files based on mode
            if mode == 'train':
                self.rgb_files = train_files
                self.spectral_files = train_spectral_files
            elif mode == 'val':
                self.rgb_files = val_files
                self.spectral_files = val_spectral_files
            else:
                raise ValueError("Mode must be 'train', 'val', or 'test'")

            folder_set = 'Train_Images'
            
        # handle spectra repetition or truncation to match images
        num_images = len(self.rgb_files)
        num_spectra = spectra_array.shape[0]
        if num_spectra < num_images:
            repeat_factor = int(np.ceil(num_images / num_spectra))
            spectra_array_repeated = np.tile(spectra_array, (repeat_factor, 1))
            spectra_array_repeated = spectra_array_repeated[:num_images]
        else:
            spectra_array_repeated = spectra_array[:num_images]
        
        # map spectra to images
        self.image_spectra_mapping = {img: spectra_array_repeated[i] for i, img in enumerate(self.rgb_files)}

        # preload and align images using multiprocessing
        args_list = [
            (rgb_dir, spectral_dir, folder_set, rgb_name, idx, self.channels, self.spectral_files, self.align)
            for idx, rgb_name in enumerate(self.rgb_files)
        ]

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_image, args_list), desc=f"Loading {mode} data", total=len(self.rgb_files)))

        self.data = results

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_image, spectral_images = self.data[idx]
        spectrum = self.image_spectra_mapping[self.rgb_files[idx]]
        spectrum = np.expand_dims(spectrum, axis=0) # [1, num_bands]
        
        # convert spectrum to tensor
        spectrum = torch.tensor(spectrum, dtype=torch.float32)

        # convert to PIL
        rgb_image = Image.fromarray(rgb_image).convert('RGB')
        spectral_images = [Image.fromarray(img) for img in spectral_images]

        # apply transformations
        if self.transform:
            rgb_image = self.transform(rgb_image)
            spectral_images = [self.transform(img) for img in spectral_images]

        # hsv segmentation
        hsv_mask = self.segment_hsv(np.array(rgb_image))

        return (rgb_image, spectrum, hsv_mask, *spectral_images)
    
    def segment_hsv(self, image):
        """
        Perform HSV segmentation and save the mask for debugging.

        Args:
            image (numpy.ndarray): RGB image in HWC or CHW format, normalized between 0 and 1.

        Returns:
            numpy.ndarray: Binary mask.
        """

        if image.shape[0] == 3 and len(image.shape) == 3:
            image = image.transpose(1, 2, 0)


        if image.shape[2] != 3:  # Not 3 channels
            raise ValueError(f"Invalid image shape for HSV segmentation: {image.shape}")

        image_rescaled = (image * 255).astype(np.uint8)
        hsv_image = cv2.cvtColor(image_rescaled, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)

        return mask

    def process_image(self, args):
        rgb_dir, spectral_dir, folder_set, rgb_name, idx, channels, spectral_files, align = args

        rgb_path = os.path.join(rgb_dir, folder_set, rgb_name)
        rgb_im = cv2.imread(rgb_path)

        # ensure the base names match
        for channel in channels:
            spectral_file = spectral_files[channel][idx]
            if os.path.splitext(rgb_name)[0] != os.path.splitext(spectral_file)[0]:
                raise ValueError(
                    f"Mismatch detected: RGB file '{rgb_name}' does not match spectral file '{spectral_file}' in channel '{channel}'"
                )

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

        # validate size consistency
        for channel, spectral_im in zip(channels, spectral_images):
            assert spectral_im.shape == rgb_resized.shape[:2], \
                f"Size mismatch: RGB {rgb_resized.shape[:2]} vs {channel} {spectral_im.shape}"

        # crop images to the center
        crop_height = int(height * self.crop_factor)
        crop_width = int(width * self.crop_factor)
        crop_size = (crop_height, crop_width)
        rgb_cropped = PotatoDataset.center_crop(rgb_resized, crop_size)
        spectral_cropped = [PotatoDataset.center_crop(img, crop_size) for img in spectral_images]

        return (rgb_cropped, spectral_cropped)

    @staticmethod
    def align_images(base_img, img_to_align):
        """
        Align two images using SIFT keypoints and RANSAC.
        
        Args:
            base_img (numpy.ndarray): The base image to align to.
            img_to_align (numpy.ndarray): The image to align.
            
        Returns:
            numpy.ndarray: The aligned image.
        """ 
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_to_align, None)
        kp2, des2 = sift.detectAndCompute(base_img, None)

        matcher = cv2.FlannBasedMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # filter good matches
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                    good.append(m)

        # estimate homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # warp image
        aligned_img = cv2.warpPerspective(img_to_align, H, (base_img.shape[1], base_img.shape[0]))
        return aligned_img
    
    @staticmethod
    def center_crop(image, crop_size):
        """
        Crop the center of an image to the given size.
        
        Args:
            image (numpy.ndarray): The input image (H x W or H x W x C).
            crop_size (tuple): The desired output size (height, width).
        
        Returns:
            numpy.ndarray: The center-cropped image.
        """
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h + crop_h, start_w:start_w + crop_w]