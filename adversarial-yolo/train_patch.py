"""
Training code for Adversarial patch training


"""
# 1. 여기서 savemat 기능 import하기 !
#############################
from scipy.io import savemat
#############################
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

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0,80, self.config).cuda()
        self.total_variation = TotalVariation().cuda()
        self.content_loss = ContentLossCalculator().cuda()
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

     

        adv_patch_cpu = self.read_image("saved_patches/600300.jpg")
        
        ####################################################################
        orig_patch_cpu = self.read_image("saved_patches/600300.jpg")
          
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
        # 2. for문 시작전에 loss들을 저장할 수 있는 빈 리스트 생성하기
        #########################################################
        tv_loss_list, ct_loss_list, det_loss_list= [], [], []
        ############################################################

        for epoch in range(n_epochs):
            ep_det_loss = 0
            #ep_nps_loss = 0
            ep_tv_loss = 0
            ##############
            #ep_sim_loss = 0
            ep_cf_loss = 0
            ep_ct_loss = 0
            #ep_svm_loss =0
            #################
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    ####################################################################
                    orig_patch = orig_patch_cpu.cuda()
                    
                    #ref_patch = ref_patch_cpu.cuda()
                    ####################################################################
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    img = p_img_batch[1, :, :,]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    #img.show()

                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    tv = self.total_variation(adv_patch)
                    ct =  self.content_loss(orig_patch, adv_patch, epoch)
                    #######################################3
                    #if epoch % 5 == 0 and epoch >= 5 :
                    #    adv_patch = 0.7*adv_patch + 0.3*orig_patch
                    ####################
                    tv_loss = tv*1.0
                    ct_loss = ct*5.00
                    det_loss = torch.mean(max_prob)

                    loss = det_loss  + torch.max(tv_loss, torch.tensor(0.1).cuda()) + ct_loss #+svm_loss #+ cf_loss 
                    #loss= det_loss  + ct_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())+ torch.max(cf_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    #ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    #################
                    #ep_sim_loss += sim_loss.detach().cpu().numpy()
                    #ep_cf_loss += cf_loss.detach().cpu().numpy()
                    ep_ct_loss += ct_loss.detach().cpu().numpy()
                    #ep_svm_loss += svm_loss.detach().cpu().numpy()
                    #################
                    #ep_loss += loss.item()
                    ep_loss +=loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        #self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        ###########################################################################################
                        #self.writer.add_scalar('loss/cf_loss', cf_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/ct_loss', ct_loss.detach().cpu().numpy(), iteration)
                        #self.writer.add_scalar('loss/svm_loss', svm_loss.detach().cpu().numpy(), iteration)
                        
                        #self.writer.add_scalar('loss/sim_loss', sim_loss.detach().cpu().numpy(), iteration)
                        ###########################################################################################
                        
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, tv_loss, loss, ct_loss #, svm_loss #cf_loss
                        ##################위에거
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()

            ep_det_loss = ep_det_loss/len(train_loader)
            #ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ##############################
            #ep_sim_loss = ep_sim_loss/len(train_loader)
            #ep_cf_loss = ep_cf_loss/len(train_loader)
            ep_ct_loss = ep_ct_loss/len(train_loader)
            #ep_svm_loss = ep_svm_loss/len(train_loader)
            #########################
            ep_loss = ep_loss/len(train_loader)
            
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #im = transforms.ToPILImage('RGB')(adv_patch)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')
            im.save(f'pics/{time_str}_epoch{epoch}.jpg')
            scheduler.step(ep_loss)

            # 3. 이미지를 저장한 이후 각 list별로 loss를 append해주고 mat 생성
            ############################################################
            det_loss_list.append(ep_det_loss)
            ct_loss_list.append(ep_ct_loss)
            tv_loss_list.append(ep_tv_loss)
            mdic = {"det_loss": det_loss_list, "ct_loss": ct_loss_list, "tv_loss": tv_loss_list}
            save_path = "/content/drive/MyDrive/adversarial-yolo/result"
            # 위의 save_path는 당신이 mat파일을 저장할 폴더로
            if os.path.exists(save_path) == False:
                os.mkdir(save_path)
            savemat(save_path + '/112001_log.mat', mdic)
            # 위의 savemat 할때 이름을 원하는 이름으로
            # 주의! 새로운 조건으로 train 할때마다 이름 바꿔주기
            # 아니면 그전 로그 덮어쓰기됨 ! 
            ############################################################
            if True:
               
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                #print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                ##################
                #print('   SIM LOSS: ', ep_sim_loss)
                #print('   CF LOSS: ', ep_cf_loss)
                print('   CT LOSS: ', ep_ct_loss)
                #print('   SVM LOSS: ', ep_svm_loss)
                
                ##################
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                
                plt.imshow(im)
                plt.show()
                
                name = '/'+ time.strftime("%Y%m%d")+'_1_6_crop_2to1_aft50.jpg'####mine
        
                im.save("saved_patches"+name) # final patch img
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, tv_loss, loss, ct_loss #, svm_loss ##추가
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
            #adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
            adv_patch_cpu = torch.full((3,300,150), 0.5)
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
        #tf=transforms.Resize(self.config.patch_size))
        #tf =transforms.Resize((self.config.patch_size, self.config.patch_size))
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

