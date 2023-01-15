from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')       
        self.parser.add_argument('--train_cls_data_source', type=bool, default=False, help='')
        self.parser.add_argument('--use_data_Tri', type=bool, default=True, help='')

        self.isTrain = True

