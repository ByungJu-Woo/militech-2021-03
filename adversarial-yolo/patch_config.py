from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        '''
        #for train
        #self.img_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Train/pos"
        #self.lab_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Train/pos/yolo-labels"
        
        #for test
        self.img_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Test/pos"
        self.lab_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Test/patched/gt_label"
        
        self.cfgfile = "/content/drive/MyDrive/adversarial-yolo/cfg/yolov2.cfg"
        self.weightfile = "/content/drive/MyDrive/adversarial-yolo/weights/yolo.weights"
        self.printfile = "/content/drive/MyDrive/adversarial-yolo/non_printability/30values.txt"
        '''

        self.img_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Train/pos"
        self.lab_dir = "/content/drive/MyDrive/custom_model_training/inria_person_data/inria/Train/pos/yolo-labels"
        self.cfgfile = "/content/drive/MyDrive/adversarial-yolo/cfg/yolov2.cfg"
        self.weightfile = "/content/drive/MyDrive/adversarial-yolo/weights/yolo.weight"
        self.printfile = "/content/drive/MyDrive/adversarial-yolo/non_printability/30values.txt"
        
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = [300,150]
        #self.patch_size = 300
        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj
}