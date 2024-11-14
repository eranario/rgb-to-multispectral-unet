import os
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from PIL import Image

class PotatoDataset(Dataset):
    
    def __init__(self, img_dir, label_dir, transform=None):
        """Multispectral Potato Detection and Classification Dataset

        Args:
            img_dir (str): Directory with all the images.
            label_dir (str): Directory with all the XML label files.
            transform (callable, optional): If specified, this transform will be applied on the image.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.img_files)
    
    def parse_xml(self, xml_file):
        """
        Parse XML file to extract bounding box coordinates and labels.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return boxes, labels
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_name).convert("RGB")
        xml_file = os.path.join(self.label_dir, os.path.splitext(self.img_files[idx])[0] + '.xml')
        
        boxes, labels = self.parse_xml(xml_file)

        if self.transform:
            image = self.transform(image)
        
        target = {'boxes': boxes, 'labels': labels}
        return image, target