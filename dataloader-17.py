from utils.imports import *
from fastai.dataset import *
from sklearn.model_selection import train_test_split

class Shanghai5120(BaseDataset):
    def __init__(self, fn_x, fn_y, transform=None):
        self.fn_x = [str(o) for o in fn_x]
        self.fn_y = [str(o) for o in fn_y]
        super().__init__(transform)

    def get_n(self):
        return len(self.fn_x)

    def get_sz(self):
        return self.transform.sz
    
    def get_c(self):
        # temporary solution
        return 8
        
    def get_x(self, i):
        return open_image(self.fn_x[i])
    
    def get_y(self, i):
        return open_image(self.fn_y[i])

    def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))
