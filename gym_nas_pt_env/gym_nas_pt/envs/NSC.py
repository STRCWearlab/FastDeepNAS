if __package__ is None or __package__ == '':
    # uses current directory visibility
    pass
else:
    # uses current package visibility
    from .config import *
import pandas as pd


class NSC:

    def __init__(self, vec):

        """ Initialize NSC encoding of a layer from a vector of 5 values """

        self.index = vec[0]
        self.type = TYPES[vec[1]]
        self.kernel_size = vec[2]
        self.pred1 = vec[3]
        self.pred2 = vec[4]
        if len(vec) > 5:
            self.n_kernels = vec[5]
        else:
            if self.type == 'Convolution':
                self.n_kernels = N_KERNELS_CONV[0]
            else:
                self.n_kernels = 0

        r = {'Type': self.type, 'Kernel_size': self.kernel_size, 'Pred1': self.pred1,
             'Pred2': self.pred2, 'N_kernels': self.n_kernels}
        n = {'Type': vec[1], 'Kernel_size': self.kernel_size, 'Pred1': self.pred1,
             'Pred2': self.pred2, 'N_kernels': self.n_kernels}

        self.repr = pd.DataFrame(r, index=[self.index])
        self.numerical = pd.DataFrame(n, index=[self.index])
        self.tuple = vec

    def valid(self):

        """ Check that the NSC code is allowed according to the search space defined in config.py """

        if self.index > MAX_INDEX:
            return False

        elif self.type not in TYPES.values():
            return False

        elif self.type == 'Convolution' and self.kernel_size not in KERNEL_SIZES_CONV:
            return False

        elif self.type == 'Convolution' and self.pred2 != 0:
            return False

        elif self.type == 'Convolution' and self.n_kernels not in N_KERNELS_CONV:
            return False

        elif self.type == 'Max Pooling' and self.kernel_size not in KERNEL_SIZES_POOL:
            return False

        elif self.type == 'Max Pooling' and self.pred2 != 0:
            return False

        elif self.type == 'Max Pooling' and self.n_kernels != 0:
            return False

        elif self.type == 'Average Pooling' and self.kernel_size not in KERNEL_SIZES_POOL:
            return False

        elif self.type == 'Average Pooling' and self.pred2 != 0:
            return False

        elif self.type == 'Average Pooling' and self.n_kernels != 0:
            return False

        elif self.type == 'Identity' and self.pred2 != 0:
            return False

        elif self.type == 'Identity' and self.kernel_size != 0:
            return False

        elif self.type == 'Elementwise Addition' and self.kernel_size != 0:
            return False

        elif self.type == 'Elementwise Addition' and self.pred1 == self.pred2:
            return False

        elif self.type == 'Concatenation' and self.kernel_size != 0:
            return False

        elif self.type == 'Concatenation' and self.pred1 == self.pred2:
            return False

        elif self.type == 'Concatenation' and self.n_kernels != 0:
            return False

        elif self.type == 'Terminal' and self.kernel_size != 0:
            return False

        elif self.type == 'Terminal' and self.pred1 != 0:
            return False

        elif self.type == 'Terminal' and self.pred2 != 0:
            return False

        elif self.type == 'Terminal' and self.n_kernels != 0:
            return False

        elif self.pred1 >= self.index:
            return False

        elif self.pred2 >= self.index:
            return False

        else:
            return True
