"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

#############################
import os
from scipy.io import savemat
import warnings
warnings.filterwarnings(action='ignore')
#############################

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        
        ##
        self.prob_extractor = MaxProbExtractor(0,80, self.config).cuda() # DET LOSS
        self.total_variation = TotalVariation().cuda() # TV LOSS
        self.MSE_calculator= MSECalculator().cuda() # MSE LOSS
        self.camouflage_calculator = CamouflageCalculator().cuda() # CF LOSS
        self.perceptual_calculator = VGGPerceptualLoss().cuda() # PC LOSS
        ##

        self.patch_name = 'PATCH B'
        self.writer = self.init_tensorboard(mode)

        

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")
     
        # Generate starting point
        # adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu = self.read_image("saved_patches/MILITARY.jpg")
        orig_patch_cpu = self.read_image("saved_patches/MILITARY.jpg")
          
        adv_patch_cpu.requires_grad_(True)
        
        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=5)
                                                   #num_workers=10
        self.epoch_length = len(train_loader)
        #print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()

        tv_loss_list, MSE_loss_list, det_loss_list, cf_loss_list, pc_loss_list = [], [], [], [], []

        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_tv_loss = 0
            ep_MSE_loss = 0
            ep_cf_loss = 0
            ep_pc_loss = 0
            ep_loss = 0
            
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    orig_patch = orig_patch_cpu.cuda()
                    
                    #ref_patch = ref_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    img = p_img_batch[1, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    #img.show()


                    #################################################
                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    tv = self.total_variation(adv_patch)
                    MSE =  self.MSE_calculator(orig_patch, adv_patch, epoch)
                    cf = self.camouflage_calculator(adv_patch, orig_patch)
                    pc = self.perceptual_calculator(adv_patch, orig_patch, epoch)
                    #################################################

                    # LOSS FOR PATCH A
                    '''
                    det_loss = torch.mean(max_prob)
                    tv_loss = tv * 1.0
                    MSE_loss = MSE * 5.0
                    loss = det_loss  + torch.max(tv_loss, torch.tensor(0.1).cuda()) + MSE_loss
                    '''

                    # LOSS FOR PATCH B
                    det_loss = torch.mean(max_prob)              
                    tv_loss = tv*1.00
                    MSE_loss = MSE * 0.50
                    cf_loss = cf * 2.50
                    pc_loss = pc * 0.01
                    loss = det_loss  + tv_loss + ct_loss +pc_loss + cf_loss 
                    
                    # epoch LOSS FOR PATCH A
                    '''
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_MSE_loss += MSE_loss.detach().cpu().numpy()
                    '''

                    # epoch LOSS FOR PATCH B
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_MSE_loss += MSE_loss.detach().cpu().numpy()
                    ep_cf_loss += cf_loss.detach().cpu().numpy()
                    ep_pc_loss = pc_loss.detach().cpu().numpy()
                    ep_loss +=loss.item()



                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        

                        # print LOSS FOR PATCH A
                        '''
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/MSE_loss', MSE_loss.detach().cpu().numpy(), iteration)
                        '''

                        # print LOSS FOR PATCH B
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/MSE_loss', MSE_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/cf_loss', cf_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/pc_loss', pc_loss.detach().cpu().numpy(), iteration)
                        

                        
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, tv_loss, loss, MSE_loss,  cf_loss, pc_loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()

            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            im.save(f'pics/{time_str}_epoch{epoch}.jpg')

            # PATCH A MATFILE
            '''
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_MSE_loss = ep_MSE_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)
            scheduler.step(ep_loss)

            det_loss_list.append(ep_det_loss)
            ct_loss_list.append(ep_ct_loss)
            tv_loss_list.append(ep_tv_loss)
            mdic = {"det_loss": det_loss_list, "ct_loss": ct_loss_list, "tv_loss": tv_loss_list}
            '''

            # PATCH B MATFILE
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_MSE_loss = ep_MSE_loss/len(train_loader)
            ep_cf_loss = ep_cf_loss/len(train_loader)
            ep_pc_loss = ep_pc_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)
            scheduler.step(ep_loss)

            det_loss_list.append(ep_det_loss)
            tv_loss_list.append(ep_tv_loss)
            MSE_loss_list.append(ep_MSE_loss)
            cf_loss_list.append(ep_cf_loss)
            pc_loss_list.append(ep_pc_loss)
            mdic = {"det_loss": det_loss_list, "tv_loss": tv_loss_list, "MSE_loss": MSE_loss_list, "cf_loss": cf_loss_list, "pc_loss":pc_loss_list}
            
            save_path = "/content/drive/MyDrive/adversarial-yolo/result"
            if os.path.exists(save_path) == False:
                os.mkdir(save_path)
            savemat(save_path + self.patch_name, mdic)
            ############################################################
            if True:
               
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('   CF LOSS: ', ep_cf_loss)
                print('   CT LOSS: ', ep_ct_loss)

                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                
                plt.imshow(im)
                plt.show()
            
                
                im.save("saved_patches/"+time.strftime("%m%d")+'_'+self.patch_name+'.jpg') # final patch img
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, tv_loss, loss, MSE_loss, cf_loss, pc_loss 
                torch.cuda.empty_cache()
              
                #sys.stdout.close()#######mine
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3,600,300), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf=transforms.Resize([600,300])

        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu



def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer('paper_obj')
    #PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()

