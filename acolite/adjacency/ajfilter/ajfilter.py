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
from skimage.transform import rescale,resize

class ajfilter():

    def __init__(self,l1r,l2r,settings):
        import os, time, datetime
        import numpy as np
        import scipy.ndimage
        import acolite as ac

        time_start = datetime.datetime.now()

        ## read gem file if NetCDF
        gem = l1r
        if type(gem) is str:
            gem = ac.gem.gem(gem)
            nc_projection = gem.nc_projection
        gemf = gem.file

        if type(l2r) is str:
            self.gem_l2r = ac.gem.gem(l2r)
            self.nc_projection_l2r = self.gem_l2r.nc_projection
        self.gemf_l2 = self.gem_l2r.file
        copyfile(self.gemf_l2, self.gemf_l2.replace(".nc", "_origin.nc"))

        ## combine default and user defined settings
        setu = ac.acolite.settings.parse(gem.gatts['sensor'], settings=settings)
        if 'verbosity' in setu: verbosity = setu['verbosity']
        if verbosity > 0: print('Initializing ajfilter for {}'.format(gemf))
        self.output_name = gem.gatts['output_name'] if 'output_name' in gem.gatts else os.path.basename(gemf).replace('.nc','_ajfilter')
        self.output_dir = os.path.split(gemf)[0]

        gem.datasets_read()
        # rhot = gem.data(ds='rhot_865')
        #
        # print(rhot.shape)
        # print()
        self.gem = gem

        sensor = gem.gatts['sensor']
        ss = sensor.lower().split('_')
        coef = ac.config['data_dir'] + '/Shared/ajfilter/{}_{}_fitting-coef.csv'.format(ss[1], ss[0])

        ## should be determined by the sensing date and image location
        atm_type = 'sas'
        coef_df = pd.read_csv(coef)
        self.coef_df = coef_df[coef_df['atm_type']=='sas']
        self.coef_cols = [col for col in self.coef_df.columns if col.startswith('coef')]
        self.offset_col = 'intercept'
        self.resolution_filter = 30 ## the resolution of filter is 30 m

        exclude_bands = []
        if sensor in ['L8_OLI']:
            self.resolution = 30
            exclude_bands = ['band_8','band_9']
        elif sensor in ['S2A_MSI', 'S2B_MSI']:
            self.resolution = settings['s2_target_res']
            exclude_bands = ['band_9', 'band_10','band_11', 'band_12']

        # maximum distance of ae is 3000m
        self.max_distance_ae = int(settings['ajfilter_max_distance'])
        self.filter_dim = int(self.max_distance_ae/self.resolution_filter)
        self.filter = self.__gen_filter()

        wave_range = settings['ajfilter_wave_range']
        rhot_adj_bands = [i for i in sorted([b for b in gem.datasets if b.startswith('rhot')]) if
                          (int(i.split('_')[1])>int(wave_range[0]))
                          and int(i.split('_')[1])<int(wave_range[1])]

        rhot_adj_bands_new = rhot_adj_bands.copy()
        self.rhot_adj_original = []
        self.rhos_difference = {}
        for b in tqdm(rhot_adj_bands,"Initializing ajfiter"):
            rhot, attr = self.gem.data(ds=b, attributes=True)
            band_name = str.lower(attr['PAR'])
            if band_name in exclude_bands:
                rhot_adj_bands_new.remove(b)
                continue
            self.rhot_adj_original.append((band_name,b,rhot,attr))

            name_rhos = b.replace("rhot","rhos")
            rhos = self.gem_l2r.data(ds=name_rhos, attributes=False)
            if self.resolution != self.resolution_filter:
                rhos_re = rescale(rhos,self.resolution/self.resolution_filter,anti_aliasing=False)
                rhos_weighted_ave_re = signal.convolve2d(rhos_re, self.filter, boundary='symm', mode='same')
                rhos_weighted_ave = resize(rhos_weighted_ave_re, (rhos.shape[0],rhos.shape[1]), anti_aliasing=False)
                print(rhos.shape,rhos_re.shape,rhos_weighted_ave.shape)
            else:
                rhos_weighted_ave = signal.convolve2d(rhos, self.filter, boundary='symm', mode='same')

            self.rhos_difference[band_name] = (rhos_weighted_ave - rhos)
        self.rhot_adj_bands = rhot_adj_bands_new
        print("adjacency correction bands:{}".format(rhot_adj_bands_new))



    def run(self,acmode,aot550,senz,iteration=0):
        desc = "Adjacency correctiong, Iteration:{}".format(iteration)
        # self.gem.datasets_read()
        iband = 1
        for b_name,ref_name,rhot_original,attr_original in tqdm(self.rhot_adj_original,desc=desc):
            if iteration == 1:
                ref_name_rhos = ref_name.replace('rhot', 'rhos')
                rhos = self.gem_l2r.data(ds=ref_name_rhos, attributes=False)

                if self.resolution != self.resolution_filter:
                    rhos_re = rescale(rhos, self.resolution / self.resolution_filter, anti_aliasing=False)
                    rhos_weighted_ave_re = signal.convolve2d(rhos_re, self.filter, boundary='symm', mode='same')
                    rhos_weighted_ave = resize(rhos_weighted_ave_re, (rhos.shape[0],rhos.shape[1]),
                                                anti_aliasing=False)
                else:
                    rhos_weighted_ave = signal.convolve2d(rhos, self.filter, boundary='symm', mode='same')

                # rhos_weighted_ave = signal.convolve2d(rhos, self.filter, boundary='symm', mode='same')
                # rho_adj = (rhos_weighted_ave - rhos) * _q
                self.rhos_difference[b_name] = (rhos_weighted_ave - rhos)


            backup_l1rname = os.path.join(self.output_dir,
                                              os.path.basename(self.gem.file).replace('.nc',
                                                                                      '_{}.nc'.format(iteration)))
            _q = self.__cal_diffuse_direct_trans_ratio(acmode=acmode, aot550=aot550, senz=senz, band_name=b_name)
            print("-----------Estimated Aerosol:{}, {},Q factor:{} for {}".format(acmode, aot550, _q, b_name))
            # rho_adj = (rhos_weighted_ave - rhos) * _q
            rho_adj = self.rhos_difference[b_name]*_q

            output_name_png = self.output_name.replace('_ajfilter', '_ajfilter.png')
            output_name_nc = self.output_name.replace('_ajfilter', '_ajfilter.nc')
            rho_t_cor = rhot_original - rho_adj
            rho_t_cor[rho_t_cor<=0] = 2e-4

            ## backup the original L1R

            copyfile(self.gem.file, backup_l1rname)
            if iteration==0:
                self.orignal_l1r = backup_l1rname

            self.gem.write(ds=ref_name, data=rho_t_cor, ds_att=attr_original)


            rhot_adj_new = True if (iteration == 0) and (iband==1)  else False
            ac.output.nc_write(os.path.join(self.output_dir, output_name_nc), '{}_adj_{}'.format(ref_name, iteration),
                               rho_adj, attributes={}, replace_nan=True,
                               new=rhot_adj_new,
                               dataset_attributes={},
                               nc_projection=self.gem.nc_projection)
            iband += 1

        return self.gem.file


    def correct_l2r(self,acmode,aot550,senz,settings):
        iband = 1
        for b_name,ref_name,rhot_original,attr_original in tqdm(self.rhot_adj_original,"correct final adj for l2r"):
            name_rhos = ref_name.replace("rhot","rhos")
            rhos,rhos_att = self.gem_l2r.data(ds=name_rhos, attributes=True)

            _q = self.__cal_diffuse_direct_trans_ratio(acmode=acmode, aot550=aot550, senz=senz, band_name=b_name)
            # print("-----------Estimated Aerosol:{}, {},Q factor:{} for {}".format(acmode, aot550, _q, b_name))
            # rho_adj = (rhos_weighted_ave - rhos) * _q
            rho_adj = self.rhos_difference[b_name]*_q

            # output_name_png = self.output_name.replace('_ajfilter', '_ajfilter.png')
            output_name_nc = self.output_name.replace('_ajfilter', '_ajfilter.nc')
            rho_s_cor = rhos - rho_adj
            rho_s_cor[rho_s_cor<=0] = 3e-4

            self.gem_l2r.write(ds=name_rhos, data=rho_s_cor, ds_att=rhos_att)
            ac.output.nc_write(os.path.join(self.output_dir, output_name_nc), '{}_adj'.format(ref_name),
                               rho_adj, attributes={}, replace_nan=True,
                               new=True,
                               dataset_attributes={},
                               nc_projection=self.gem.nc_projection)
            iband += 1

        return self.gem_l2r.file,settings


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
        coef_df = self.coef_df[(self.coef_df['aero_type']==acmode) & (self.coef_df['band_name']==band_name)]
        # features['x*y'] = x*y
        # features['x*y^2'] = np.multiply(x,y**2)
        # features['x^2*y^0'] = np.multiply(x**2, y**0)
        # features['x^2*y'] = np.multiply(x**2, y)
        # features['x^3*y^2'] = np.multiply(x**3, y**2)
        # features['x^3*y'] = np.multiply(x**3, y)
        # features['x^0*y^3'] = np.multiply(x**0, y**3)

        coef = coef_df[self.coef_cols].values[0]
        offset = coef_df[self.offset_col].values[0]
        x = 1.0 if aot550>0.5 else aot550/0.5
        y = 1.0 if senz>60 else senz/60.0

        ratio = np.sum(coef*np.array([x*y,x*y**2,x**2,x**2*y,x**3*y**2,x**3*y,y**3]))+offset
        return ratio
