import acolite as ac
import numpy as np
from scipy import optimize
import sys, os, time
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from tqdm import tqdm

class ajfilter_v2():

    def __init__(self,l2r,settings):
        import os, time, datetime
        import numpy as np
        import scipy.ndimage
        import acolite as ac

        time_start = datetime.datetime.now()

        if type(l2r) is str:
            self.gem_l2r = ac.gem.gem(l2r)
            self.nc_projection_l2r = self.gem_l2r.nc_projection
        self.gemf_l2 = self.gem_l2r.file

        self.output_name = self.gem_l2r.gatts['output_name'] if 'output_name' in \
                                                                self.gem_l2r.gatts \
            else os.path.basename(self.gemf_l2).replace('.nc', '_ajfilter')
        self.output_dir = os.path.split(self.gemf_l2)[0]

        sensor = self.gem_l2r.gatts['sensor']
        ss = sensor.lower().split('_')
        coef = ac.config['data_dir'] + '/Shared/ajfilter/{}_{}_fitting-coef.csv'.format(ss[1], ss[0])

        ## should be determined by the sensing date and image location
        self.atm_type = 'sas'
        coef_df = pd.read_csv(coef)
        self.coef_df = coef_df[coef_df['atm_type']==self.atm_type]
        self.coef_cols = [col for col in self.coef_df.columns if col.startswith('coef')]
        self.offset_col = 'intercept'

        exclude_bands = []
        if sensor in ['L8_OLI']:
            self.resolution = 30
            self.rhos_band_dic = {'rhos_443':'band_1',
                                  'rhos_483':'band_2',
                                  'rhos_561':'band_3',
                                  'rhos_655':'band_4',
                                  'rhos_865':'band_5',
                                  'rhos_1609':'band_6',
                                  'rhos_2201':'band_7',
                                  'rhos_1373':'band_8',
                                  'rhos_592':'band_9'}
            exclude_bands = ['band_8','band_9']

        elif sensor in ['S2A_MSI', 'S2B_MSI']:
            self.resolution = settings['s2_target_res']
            self.rhos_band_dic = {'rhos_443':'band_1',
                                  'rhos_483':'band_2',
                                  'rhos_561':'band_3',
                                  'rhos_655':'band_4',
                                  'rhos_865':'band_5',
                                  'rhos_1609':'band_6',
                                  'rhos_2201':'band_7',
                                  'rhos_1373':'band_8'}

        # maximum distance of ae, defult is 3000m
        if 'ajfilter_max_distance' not in settings:
            settings['ajfilter_max_distance'] = 3e3
        self.max_distance_ae = int(settings['ajfilter_max_distance'])
        self.filter_dim = int(self.max_distance_ae/self.resolution)
        self.filter = self.__gen_filter()
        # wave_range = (400,2300)
        self.wave_range = settings['ajfilter_wave_range']

        self.gem_l2r.datasets_read()
        rhos_adj_bands = [i for i in sorted([b for b in self.gem_l2r.datasets if b.startswith('rhos')]) if
                          (int(i.split('_')[1]) > int(self.wave_range[0])
                           and int(i.split('_')[1]) < int(self.wave_range[1]))]

        rhos_adj_bands_new = rhos_adj_bands.copy()
        self.rhos_original = []
        for b in rhos_adj_bands:
            rhos, attr = self.gem_l2r.data(ds=b, attributes=True)
            # band_name = str.lower(attr['PAR'])
            if b not in self.rhos_band_dic:
                continue
            band_name = self.rhos_band_dic[b]
            if band_name in exclude_bands:
                rhos_adj_bands_new.remove(b)
                continue
            self.rhos_original.append([band_name,b,rhos,attr])

        print("adjacency correction bands:{}".format(rhos_adj_bands_new))



    def run(self,acmode,aot550,senz,max_iter=0):
        self.gem_l2r.datasets_read()
        i_iter,rhos = 0, self.rhos_original.copy()
        bandnames = list(np.asarray(self.rhos_original)[:,0])
        q_dic = self.__cal_diffuse_direct_trans_ratio(acmode=acmode, aot550=aot550, senz=senz, band_name=bandnames)
        output_name_nc = self.output_name.replace('_ajfilter', '_ajfilter.nc')

        while i_iter < max_iter:
            iband = 1
            for i in range(len(rhos)):
                (b_name, ref_name, rhos_single, attr_original) = rhos[i]

                desc = "Adjacency correctiong, Iteration:{}".format(i_iter)
                print(desc+',{}'.format(b_name))
                rhos_weighted_ave = signal.convolve2d(rhos_single, self.filter, boundary='symm', mode='same')
                rho_adj = (rhos_weighted_ave - rhos_single) * q_dic[b_name]

                rhot_adj_new = True if (i_iter == 0) and (iband==1)  else False
                ac.output.nc_write(os.path.join(self.output_dir, output_name_nc), '{}_adj_{}'.format(ref_name, i_iter),
                               rho_adj, attributes={}, replace_nan=True,
                               new=rhot_adj_new,
                               dataset_attributes={},
                               nc_projection=self.gem_l2r.nc_projection)
                # rhos_single_cor = rhos_single-rho_adj
                rhos_single_cor = self.rhos_original[i][2]-rho_adj
                rhos_single_cor[rhos_single_cor <= 0] = 1e-4

                print("rhos {} diff:{}".format(b_name,np.mean(np.abs(rhos_single_cor - rhos_single))))
                rhos[i][2] = rhos_single_cor
                iband += 1
            i_iter += 1
        print("writing result after ajfilter correction....")
        for i in range(len(rhos)):
            (b_name, ref_name, rhos_single, attr_original) = rhos[i]
            self.gem_l2r.write(ds=ref_name, data=rhos_single, ds_att=attr_original)

        return self.gemf_l2


    def __gen_filter(self,scale=1.0):
        '''
        create filter based on exponent decreasing exp(-dis*scale*1e-3)
        Args:
            resolution: spatial resolution in unit of meter
            ad_dis: the maximum distance in unit of kilometer

        Returns: filter

        '''
        filter_ = np.ones((self.filter_dim * 2 + 1, self.filter_dim * 2 + 1), dtype=float)
        center_x, center_y = self.filter_dim, self.filter_dim
        print("filter shape:{}".format((self.filter_dim * 2 + 1, self.filter_dim * 2 + 1)))
        for i in range(self.filter_dim + 1):
            for j in range(self.filter_dim + 1):
                dis = np.sqrt(((i - center_x) * self.resolution) ** 2 + ((j - center_y) * self.resolution) ** 2)
                if dis > self.max_distance_ae:
                    filter_[i, j] = 0
                else:
                    filter_[i, j] = np.exp(-dis*scale*1e-3)

        filter_[0:self.filter_dim + 1, self.filter_dim + 1:] = filter_[0:self.filter_dim + 1, 0:self.filter_dim][:, ::-1]

        filter_[self.filter_dim + 1:, self.filter_dim + 1:] = filter_[0:self.filter_dim, 0:self.filter_dim][::-1, ::-1]
        filter_[self.filter_dim + 1:, 0:self.filter_dim + 1] = filter_[0:self.filter_dim, 0:self.filter_dim + 1][::-1, :]
        return filter_ / np.sum(filter_)


    def __cal_diffuse_direct_trans_ratio(self,acmode,aot550,senz,band_name):
        # features['x*y'] = x*y
        # features['x*y^2'] = np.multiply(x,y**2)
        # features['x^2*y^0'] = np.multiply(x**2, y**0)
        # features['x^2*y'] = np.multiply(x**2, y)
        # features['x^3*y^2'] = np.multiply(x**3, y**2)
        # features['x^3*y'] = np.multiply(x**3, y)
        # features['x^0*y^3'] = np.multiply(x**0, y**3)
        ratio_dic = {}
        if type(band_name) is str:
            band_name = [band_name]

        for b in band_name:
            coef_df = self.coef_df[(self.coef_df['aero_type']==acmode) & (self.coef_df['band_name']==b)]
            coef = coef_df[self.coef_cols].values[0]
            offset = coef_df[self.offset_col].values[0]
            x = 1.0 if aot550>0.5 else aot550/0.5
            y = 1.0 if senz>60 else senz/60.0
            ratio = np.sum(coef*np.array([x*y,x*y**2,x**2,x**2*y,x**3*y**2,x**3*y,y**3]))+offset
            ratio_dic[b] = ratio
        return ratio_dic
