import os

class Parameters:
    def __init__(self):
        self.base_dir = './'

        # folderul parinte pentru antrenare
        self.dir_pos_examples = os.path.join(self.base_dir, 'antrenare')

        # folderele reale cu imagini pozitive
        self.dirs_pos_examples = [
            os.path.join(self.base_dir, "antrenare", "daphne"),
            os.path.join(self.base_dir, "antrenare", "fred"),
            os.path.join(self.base_dir, "antrenare", "shaggy"),
            os.path.join(self.base_dir, "antrenare", "velma")
        ]

        self.dir_neg_examples = os.path.join(self.base_dir, 'negative')
        self.dir_test_examples = os.path.join(self.base_dir, 'validare/validare')
        self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare.txt')

        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # parametri
        self.dim_window = 36
        self.dim_hog_cell = 6
        self.dim_descriptor_cell = 36
        self.number_positive_examples = 0
        self.number_negative_examples = 30000
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 4
        self.use_hard_mining = False
