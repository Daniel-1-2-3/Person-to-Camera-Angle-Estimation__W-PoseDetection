import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset
from torch.quantization import get_default_qconfig, QConfig
from torch.quantization import default_observer
from torch.utils.data import Dataset, DataLoader 


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image.unsqueeze(0)

#load MiDaS model
model = torch.hub.load('intel-isl/MiDaS','MiDaS_small')
model.eval()  #set the model to evaluation mode

#define the transformations to be applied to the images (e.g., resize, normalization)
transform = transforms.Compose([
    transforms.Resize((384, 640)),
    transforms.ToTensor(),
])

#load the calibration data
calibration_data = CustomImageDataset(img_dir='C:\\Daniel\\Python\\AutoAim\\ReDWeb_V1\\Imgs', transform=transform)
calibration_loader = DataLoader(calibration_data, batch_size=1, shuffle=True)

#prepare the model for quantization, create a qconfig
#qconfig specifies what type of observer used. Observer used to calculate scale and zero point during calibration. 
#zero point: the quantized integer value that corresponds to the floating point value of 0.0
#scale: scale down the floating point values to fit into the range specified for the quantized weight values

qconfig = torch.quantization.get_default_qat_qconfig('qnnpack') #!!!

#apply the qconfig to ConvTranspose layers
model.qconfig = qconfig
prepared_model = torch.quantization.prepare(model, inplace=False)

# Calibrate the model
for data in tqdm(calibration_loader, desc="Calibration Progress"):
    with torch.no_grad():
        output = prepared_model(data)

# Convert to quantized model
quantized_model = torch.quantization.convert(prepared_model, inplace=False)
torch.save(quantized_model, 'quantized_midas.pth')
