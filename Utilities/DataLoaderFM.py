import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from Utilities.CUDA_Check import GPUorCPU
from torchvision.io import read_image, ImageReadMode


DEVICE = GPUorCPU().DEVICE
model_input_image_size_height = 256
model_input_image_size_width = 256
random_crop_size = 224

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class DataLoader_Train(Dataset):
    train_valid_transforms = transforms.Compose(
        [
            # transforms.CenterCrop(224),
            transforms.Resize((model_input_image_size_height, model_input_image_size_width), antialias=False),
            transforms.RandomCrop(random_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ZeroOneNormalize(),
        ]
    )

    train_valid_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B, file_list_GT, file_list_DM, file_list_Edge):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.file_list_GT = file_list_GT
        self.file_list_DM = file_list_DM
        self.file_list_Edge = file_list_Edge
        self.transform1 = self.train_valid_transforms
        self.transform2 = self.train_valid_transforms_Norm

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT) == len(self.file_list_DM):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        seed = torch.random.seed()

        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_A = self.transform1(img_A)
        imgA_transformed = self.transform2(img_A)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_B = self.transform1(img_B)
        imgB_transformed = self.transform2(img_B)

        imgGT_path = self.file_list_GT[idx]
        img_GT = read_image(imgGT_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        imgGT_transformed = self.transform1(img_GT)

        imgDM_path = self.file_list_DM[idx]
        img_DM = read_image(imgDM_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgDM_transformed = self.transform1(img_DM)

        imgEdge_path = self.file_list_Edge[idx]
        img_Edge = read_image(imgEdge_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgEdge_transformed = self.transform1(img_Edge)

        return imgA_transformed, imgB_transformed, imgGT_transformed, imgDM_transformed, imgEdge_transformed


class Dataloader_Eval(Dataset):
    eval_transforms = transforms.Compose(
        [
            # transforms.Resize((520, 520)),
            # transforms.ToTensor(),
            ZeroOneNormalize(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform1 = self.eval_transforms
        self.transform2 = self.eval_transforms

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        # img_A = Image.open(imgA_path).convert('RGB')
        imgA_transformed = self.transform1(img_A).to(DEVICE)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        # img_B = Image.open(imgB_path).convert('RGB')
        imgB_transformed = self.transform1(img_B).to(DEVICE)

        return imgA_transformed, imgB_transformed
