import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
####################################
import cv2
import torchvision
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
#############################
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet import Darknet

from median_pool import MedianPool2d

print('starting test read')
im = Image.open('data/horse.jpg').convert('RGB')
print('img read!')

import patch_config 

# DET LOSS
# USED FOR BOTH PATCH A, PATCH B
class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf




# TV LOSS
# USED FOR BOTH PATCH A, PATCH B
class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


# MSE LOSS
# USED FOR BOTH PATCH A, PATCH B
class MSECalculator(nn.Module):
    def __init__(self):
        super(MSECalculator, self).__init__()
        
    def forward(self, orig_patch, adv_patch, epoch):
        return torch.mean((orig_patch-adv_patch)**2)



# CF LOSS
# USED FOR PATCH B
class CamouflageCalculator(nn.Module):
    """CamouflageCalculator: calculates difference between adv_patch and orig_patch
    """

    def __init__(self):
        super(CamouflageCalculator, self).__init__()

    def forward(self, adv_patch, orig_patch):
        pls = transforms.Compose([
          transforms.RandomCrop((600,300)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
        ])
        img0 = Image.open("/content/drive/MyDrive/adversarial-yolo/saved_patches/marpat_top_crop_2.jpg")
        img1 = pls(img0)
        ref = img1.cuda()
        red_diff = torch.abs((adv_patch[0,:,:]-ref[0,:,:]+0.000001)**2)
        green_diff = torch.abs((adv_patch[1,:,:]-ref[1,:,:]+0.000001)**2)
        blue_diff = torch.abs((adv_patch[2,:,:]-ref[2,:,:]+0.000001)**2)
        diff = 0.4*red_diff + 0.1*green_diff + 0.3*blue_diff 
        diff = torch.sum(torch.sum(diff,0),0)
        return diff/torch.numel(adv_patch)


# PC LOSS
# USED FOR PATCH B
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, epoch, feature_layers=[0, 1, 2, 3], style_layers=[],):
        input = torch.unsqueeze(input, dim=0)
        
        pls = transforms.Compose([
          transforms.RandomCrop((600,300)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
        ])
        img0 = Image.open("/content/drive/MyDrive/adversarial-yolo/saved_patches/marpat_top_crop_2.jpg")
        img1 = pls(img0)
        ref = img1.cuda()

        if (epoch%10 == 9):
            target = torch.unsqueeze(ref, dim=0)
        else :
            target = torch.unsqueeze(target, dim=0)

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7,same=True)
        

        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''
        
        
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        pad_y = (img_size - adv_patch.size(-2)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)#.unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        
        # Contrast, brightness and noise transforms
        
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()


        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()


        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor


        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        
        #mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad_y + 0.5), int(pad_y)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)


        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else: 
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if(rand_loc):
            off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
            target_x = target_x + off_x
            off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            target_y = target_y + off_y
        target_y = target_y - 0.05
        ##############
        scale = target_size / current_patch_size
        #scale = target_size / (1.30*current_patch_size)
        ###############
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])


        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2
        sin = torch.sin(angle)
        cos = torch.cos(angle)        

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)


        
        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        return adv_batch_t * msk_batch_t

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch



class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        img_names = self.img_names[idx]
        # FOR TRAIN
        #return image, label
        # FOR TEST
        return image, label, img_names

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

if __name__ == '__main__':
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()

    test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                              batch_size=3, shuffle=True)

    cfgfile = "/content/drive/MyDrive/adversarial-yolo/cfg/yolov2.cfg" #wrong does not effect error -> not use??
    weightfile = "/content/drive/MyDrive/adversarial-yolo/weights/yolo.weights"
    printfile = "/content/drive/MyDrive/adversarial-yolo/non_printability/30values.txt"

    patch_size = 400
    patchfile = patch_config.ReproducePaperObj()
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    #prob_extractor = MaxProbExtractor(0,1, patchfile).cuda()
    prob_extractor = MaxProbExtractor(0,80, patchfile).cuda()
    nms_calculator = NMSCalculator(printfile, patch_size)
    total_variation = TotalVariation()
    '''
    img = Image.open('data/horse.jpg').convert('RGB')
    img = img.resize((darknet_model.width, darknet_model.height))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = torch.autograd.Variable(img)

    output = darknet_model(img)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    
    tl0 = time.time()
    tl1 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        tl1 = time.time()
        print('time to fetch items: ',tl1-tl0)
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch = Image.open('data/horse.jpg').convert('RGB')
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        img_size = img_batch.size(-1)
        print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer.forward(adv_patch, lab_batch, img_size)
        print('applying patches')
        t1 = time.time()
        img_batch = patch_applier.forward(img_batch, adv_batch_t)
        img_batch = torch.autograd.Variable(img_batch)
        img_batch = F.interpolate(img_batch,(darknet_model.height, darknet_model.width))
        print('running patched images through model')
        t2 = time.time()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    try:
                        print(type(obj), obj.size())
                    except:
                        pass
            except:
                pass

        print(torch.cuda.memory_allocated())

        output = darknet_model(img_batch)
        print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        t4 = time.time()
        nms = nms_calculator.forward(adv_patch)
        tv = total_variation(adv_patch)
        print('---------------------------------')
        print('        patch transformation : %f' % (t1-t0))
        print('           patch application : %f' % (t2-t1))
        print('             darknet forward : %f' % (t3-t2))
        print('      probability extraction : %f' % (t4-t3))
        print('---------------------------------')
        print('          total forward pass : %f' % (t4-t0))
        del img_batch, lab_batch, adv_patch, adv_batch_t, output, max_prob
        torch.cuda.empty_cache()
        tl0 = time.time()
