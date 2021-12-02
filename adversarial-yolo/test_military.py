from load_data import *
from torch import autograd
from torchvision import transforms
import patch_config
from utils import *
import matplotlib.pyplot as plt
from tqdm import *

from PIL import Image, ImageOps


class PatchTrainer(object):
    def __init__(self, mode, pic_name):
        self.config = patch_config.patch_configs[mode]()
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.region_loss = self.darknet_model.loss
       
        self.patch_applier = PatchApplier().cuda()
        
        self.patch_transformer = PatchTransformer().cuda()
        self.darknet_model = self.darknet_model.eval().cuda()  # TODO: Why eval?
    
        self.prob_extractor = MaxProbExtractor(0,80, self.config).cuda()
        #1
        #self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        ###########################################################################
        self.content_loss = ContentLossCalculator().cuda()
        #self.camouflage_calculator = CamouflageCalculator().cuda()
        
        self.pic_name = pic_name
        ###########################################################################
        #self.white_total_variation = WhiteTotalVariation().cuda()

        # self.writer = self.init_tensorboard(mode)

    def detect(self, weightfile, image_dir, save_root):
        m = self.darknet_model

        m.print_network()
        m.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))
        
        if m.num_classes == 20:
            namesfile = 'data/voc.names'
        elif m.num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'
        '''
        namesfile = 'data/coco.names'
        '''
        use_cuda = 1
        if use_cuda:
            m.cuda()

        image_list = os.listdir(image_dir)

        for i in range(len(image_list)):
            image_path = os.path.join(image_dir, image_list[i])
            img = Image.open(image_path).convert('RGB')
            sized = img.resize((m.width, m.height))

            ##### detect

            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            
            #class_names = load_class_names(namesfile)
            ############################################################
            class_names = load_class_names('data/coco.names')
            ############################################################

            if os.path.exists(save_root) == False:
                os.mkdir(save_root)
            save_path = os.path.join(save_root, image_list[i])
            plot_boxes(img, boxes, save_path, class_names)

    def image_save(self, save_path):

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        max_lab = 14
 
        test_img_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Test/pos"
        test_lab_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Test/pos/gt_label"
        '''
        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir,
                                                                max_lab, img_size, shuffle=False),
                                                   batch_size=8,
                                                   shuffle=False,
                                                   num_workers=10)#10 to 2
        '''
        train_loader = torch.utils.data.DataLoader(InriaDataset(test_img_dir,test_lab_dir,
                                                                max_lab, img_size, shuffle=False),
                                                   batch_size=8,
                                                   shuffle=False,
                                                   num_workers=2)
        #img_batch, lab_batch = next(iter(train_loader)) #img_names 뺌
        self.epoch_length = len(train_loader)

        patch_img = Image.open("/content/drive/MyDrive/adversarial-yolo/result/patches/"+self.pic_name+".jpg").convert('RGB')    #### generated patch use
        tf = transforms.Resize((300, 300))##
        #tf = transforms.Resize((300, 300))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        adv_patch = adv_patch_cpu.cuda()
        for i_batch,(img_batch, lab_batch, img_names) in tqdm(enumerate(train_loader),desc=f'Running epoch {1}', total=self.epoch_length):
            
            img_batch = img_batch.cuda()
            lab_batch = lab_batch.cuda()
            adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=False,
                                                 rand_loc=False)
            p_img_batch = self.patch_applier(img_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

            for i in range(img_batch.size(0)):
                img = p_img_batch[i, :, :, ]
                img = transforms.ToPILImage()(img.detach().cpu())
                image_name = img_names[i].replace('.png', '.png')
                new_save_path = os.path.join(save_path, image_name)

                img.save(new_save_path)


def main():
    print('this is',str(sys.argv[2]))
    trainer = PatchTrainer("paper_obj", str(sys.argv[2]))
    
    _weightfile = "/content/drive/MyDrive/adversarial-yolo/weights/yolo.weight"
    _image_dir = "/content/drive/MyDrive/adversarial-yolo/result/test_military"
    _save_root = "/content/drive/MyDrive/adversarial-yolo/result/test_military_detection"

    
    trainer.image_save(save_path=_image_dir)
    trainer.detect( weightfile = _weightfile,                 #### detection result with an adversarial patch
                   image_dir=_image_dir,
                   save_root=_save_root)
    ###


if __name__ == '__main__':
    main()

