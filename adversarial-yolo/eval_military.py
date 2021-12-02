
from tqdm import *
from load_data import *
from utils import *
from torch import autograd
from torchvision import transforms
from sklearn import metrics
import numpy as np
import patch_config
import sys
from scipy.io import savemat

image_root = "/content/drive/MyDrive/adversarial-yolo/result/test_military"                               #### image list with an adversarial patch
label_root =  "/content/drive/MyDrive/adversarial-yolo/result/new_label"                              #### new label for square padding

class Evaluator(object):
    def __init__(self, mode, save_name):
        self.config = patch_config.patch_configs[mode]()
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()
        ######################
        self.save_name=save_name
        ######################
    def evaluation(self):
        def truths_length(truths):
            for i in range(14):
                if truths[i][0] == 1:
                    return i

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        max_lab = 14
        num_classes = self.darknet_model.num_classes
        anchors = self.darknet_model.anchors
        num_anchors = self.darknet_model.num_anchors
        conf_thresh = 0.1 
        nms_thresh = 0.4
        iou_thresh = 0.5
        eps = 1e-5


        train_loader = torch.utils.data.DataLoader(InriaDataset(image_root, label_root, max_lab, img_size, shuffle=False),batch_size=8,shuffle=False, num_workers=10)
        #precision = np.zeros(np.size(np.arange(0.05,0.99,0.15)))
        #recall = np.zeros(np.size(np.arange(0.05,0.99,0.15)))
        #fscore = np.zeros(np.size(np.arange(0.05,0.99,0.15)))
        precision = np.zeros(np.size(np.arange(0.05,0.95,0.02)))
        recall = np.zeros(np.size(np.arange(0.05,0.95,0.02)))
        fscore = np.zeros(np.size(np.arange(0.05,0.95,0.02)))
        cnt = 0
        #for t in np.arange(0.05,0.99,0.15):
        for t in np.arange(0.05,0.95,0.02):
            total = 0.0
            proposals = 0.0
            correct = 0.0

            for i_batch, (img_batch, lab_batch, name) in tqdm(enumerate(train_loader)):

                img_batch = img_batch.cuda()
                lab_batch = lab_batch
                #print('name is ', name[0])
                output = self.darknet_model(img_batch)

                output=output.data
                all_boxes = get_region_boxes(output, t, num_classes, anchors, num_anchors)

                # print(name)
                for i in range(output.size(0)):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    truths = lab_batch[i].view(-1, 5)
                    num_gts = truths_length(truths)

                    total = total + num_gts

                    for i in range(len(boxes)):
                      #14
                        if boxes[i][4] > t and boxes[i][6]==0:
                            proposals = proposals + 1

                    for i in range(num_gts):
                        box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                        best_iou = 0
                        best_j = -1
                        for j in range(len(boxes)):
                            iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                            if iou > best_iou:
                                best_j = j
                                best_iou = iou
                        if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                            correct = correct + 1
            precision[cnt] = 1.0 * correct / (proposals + eps)
            recall[cnt] = 1.0 * correct / (total + eps)
            fscore[cnt] = 2.0 * precision[cnt] * recall[cnt] / (precision[cnt] + recall[cnt] + eps)
            precision=np.clip(precision,0,1)
            recall=np.clip(recall, 0, 1)
            fscore=np.clip(fscore, 0, 1)
            logging("precision: %f, recall: %f, fscore: %f" % (precision[cnt], recall[cnt], fscore[cnt]))
            cnt += 1
        auc = metrics.auc(recall, precision)
        mdic = {"recall": recall, "precision": precision, "fscore": fscore, "AUC": auc}
        save_path = "/content/drive/MyDrive/adversarial-yolo/result"                  ##### AP result is saved with a MATLAB file
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        name = '/'+self.save_name+'_AP.mat'
        #name = '/'+'standard_military_AP.mat'
        # '/{time_str}_AP.mat'
        #print('what? ',save_path+name)
        print("save name is", name)
        savemat(save_path+name, mdic)
        print (auc)


def main():

    tester = Evaluator('paper_obj', sys.argv[2])
    tester.evaluation()

if __name__ == '__main__':
    main()