import numpy as np
from scipy.stats import weibull_max
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
import KDEpy
import sklearn
import math
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity


class KDEBandwidthSelector:
    def _init_(self, data) -> None:
        X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)
        self.train = X_train
        self.test = X_test

    def is_normal_dist(self, significance=0.05):
        d, p_value = stats.kstest(self.train, 'norm')
        if p_value <= significance:
            return False
        return True

    def bw_silverman_method(self):
        g_kde = stats.gaussian_kde(dataset=self.train, bw_method='silverman')
        return g_kde.factor

    def bw_scott(self):
        g_kde = stats.gaussian_kde(dataset=self.train, bw_method='scott')
        return g_kde.factor

    def bw_sheather_jones(self):
        kdepy = KDEpy.FFTKDE(kernel='gaussian', bw='ISJ')
        if len(self.train.shape) != 2:
            if len(self.train.shape) == 1:
                train_data = np.reshape(self.train, (len(self.train), 1))
                bw = kdepy.bw_method(train_data)
                return bw
            else:
                raise Exception('Incorrect input dimension ')
        else:
            return kdepy.bw_method(self.train)
        # x,y = KDEpy.FFTKDE(kernel=kernal, bw= 'ISJ').fit(self.train).evaluate()
        # return x,y # check if we can return bw

    def bw_mlcv(self):
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(self.train[:, None]);
        return grid.best_params_['bandwidth']

    def find_bw(self):
        bw_values_dict = {}
        bw_values_dict['silverman'] = self.bw_silverman_method()
        bw_values_dict['scott'] = self.bw_scott()
        bw_values_dict['isj'] = self.bw_sheather_jones()
        bw_values_dict['mlcv'] = self.bw_mlcv()
        # if not self.is_normal_dist():
        # silverman method works well only on normally distributed data
        # del bw_values_dict['silverman']
        opt_bw = None
        opt_d_value = math.inf
        for key in bw_values_dict.keys():
            g_kde = stats.gaussian_kde(dataset=self.test, bw_method=bw_values_dict[key])
            cdf = lambda ary: np.array([g_kde.integrate_box_1d(-np.inf, x) for x in ary])
            d, p_val = stats.kstest(self.test, cdf)
            print(f'{key}: D = {d},p_value = {p_val}')
            if p_val > 0.05:
                if d < opt_d_value:
                    opt_d_value = d
                    opt_bw = key
        return bw_values_dict[opt_bw]
