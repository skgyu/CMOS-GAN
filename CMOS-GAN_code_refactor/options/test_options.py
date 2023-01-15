from .base_options import BaseOptions

class TestOptions(BaseOptions):
    
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--save_step', type=str, default='', help='save_step')
 
        self.isTrain = False

        


