"""Collection of pyeslib python functions

List of Python functions (alphabetical order):
- SeparateOnSeveralLines
- ChangeTimeScale
- ComputeStatistics
- ConvertUnits
- DefineLonLatRegularGrid
- GetGHOSTData
- GetGHOSTDataMonth
- GetLonLatGridFromReducedMONARCH
- GetRegionInformationOnGrid
- GetRegionInformationOnGridIx
- GetType
- LocateStationsOnGrid
- MemoryUsage
- PrepareTimeArrays
- SelectGHOSTStations
- SelectGHOSTStationsMonth
"""

pyeslib_data_dir = '/esarchive/scratch/hpetetin/output/pyeslib_data'

# useful dictionnaries
ccaa_dict = {'ESP':'ESP','Andalucía':'AND', 'Aragón':'ARA', 'Principado de Asturias':'AST',
             'Illes Balears':'BAL',
             'Cantabria':'CAN','Cataluña':'CAT', 'Ciudad de Ceuta':'CEU', 'Castilla-La Mancha':'ClM', 'Castilla y León':'CyL',
             'Extremadura':'EXT', 'Galicia':'GAL',
             'Comunidad de Madrid':'MAD', 'Ciudad de Melilla':'MEL', 'Región de Murcia':'MUR', 'Comunidad Foral de Navarra':'NAV',
             'La Rioja':'RIO', 'País Vasco':'PV', 'Comunitat Valenciana':'VAL'}

#'Comunidad de Madrid':'MAD', 'Comunitat Valenciana':'VAL', 'Extremadura':'EXT',
#             'Galicia':'GAL', 'Illes Balears':'BAL', 'La Rioja':'RIO', 'País Vasco':'PV', #'Melilla':'MEL','Ceuta':'CEU',
#             'Principado de Asturias':'AST', 'Región de Murcia':'MUR',
#             'Ciudad de Ceuta':'CEU', 'Ciudad de Melilla':'MEL', }

pol_dict = {'sconco3':['O3','O$_3$','ppbv','ppbv'], 'sconcno2':['NO2','NO$_2$','ppbv','ppbv'],
            'sconcform':['HCHO','HCHO','ppbv','ppbv'],'nmvoc':['NMVOC','NMVOC','ppbv','ppbv'],
            'aemisNOx':['AE[NOx]','AE[NOX]','kg/m2/s','kg/m$^2$/s'], 'aemisVOC':['AE[NMVOC]','AE[NMVOC]','kg/m2/s','kg/m$^2$/s'],
            'trcno2':['TrC-NO2','TrC-NO$_2$','Pmolec/cm2','Pmolec/cm$^2$'],
            'trcno2_AMFmonarch':['TrC-NO2','TrC-NO$_2$','Pmolec/cm2','Pmolec/cm$^2$'],
            'trcno2_pal_AMFmonarch':['TrC-NO2','TrC-NO$_2$','Pmolec/cm2','Pmolec/cm$^2$'],
            'trcno2_rpro_AMFmonarch':['TrC-NO2','TrC-NO$_2$','Pmolec/cm2','Pmolec/cm$^2$'],
            'trcno2_offl_AMFmonarch':['TrC-NO2','TrC-NO$_2$','Pmolec/cm2','Pmolec/cm$^2$'],
            'trcform':['TrC-HCHO','TrC-HCHO','Pmolec/cm2','Pmolec/cm$^2$'],
            'trcform_AMFtm5':['TrC-HCHO','TrC-HCHO','Pmolec/cm2','Pmolec/cm$^2$'],
            'trc_ratio':['TrC-(HCHO/NO2)','TrC-(HCHO/NO2)','unitless','unitless'],
            'trcratio':['TrC-(HCHO/NO2)','TrC-(HCHO/NO2)','unitless','unitless'],
            't2':['T','T','°C','°C'],
            'ws10':['WS10','WS10','m/s','m/s'],
            'wd10':['WD10','WD10','°','°'],
            'tas':['T','T','°C','°C'],'g500':['g500','g500','dam','dam'],
            'sfcWind':['WS','WS','m/s','m/s'], 'psl':['PS','PS','hPa','hPa'], 'rsds':['RSDS','RSDS','W/m2','W/m$^2$']}
exp_dict = {'a5h1':'BaseCase','a5hj':'Kz:-50%',
            'a5h5':'AE:x2',
            'a5ii':'BE-NO:x10','a5hg':'BE:x2',
            'a5mv':'PhotoloysisALL:-20%','a5mw':'PhotoloysisNO2:-20%',
            'a5ig':'#levels:48','a5hi':'OperatorSplitting',
            'a5h2':'AE-NOx:-25%','a5h3':'AE-NMVOC:-25%'} 



# useful lists
dow_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dowshortcap_list = ['MON','TUE','WED','THU','FRI','SAT','SUN']
dowshort_list = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
month_list = ['January','February','March','April','May','June','July','August','September','Octobre','November','December']
monthshortcap_list = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
monthshort_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
latlon_cities = {'Madrid'    : [40.41678,-3.70379],
                 'Barcelona' : [41.38879, 2.15899],
                 'Valencia'  : [39.46975, -0.37739],
                 'Sevilla'   : [37.38283,-5.97317],
                 'Zaragoza'  : [41.65606, -0.87734],
                 'Malaga'    : [36.72016, -4.42034],
                 'Murcia'    : [37.98704, -1.13004],
                 'PalmaDeMallorca' : [39.571625, 2.650544],
                 'Palma de Mallorca' : [39.571625, 2.650544],                 
                 'Bilbao'    : [43.26299,-2.93501],
                 'Lisbon'    : [38.718500175602045, -9.137746360840811],
                 'Granada' : [37.184674711341955, -3.6012937562023573],
                 'León':[42.597774753842444, -5.569178664508812],
                 'Alicante':[38.34338153672178, -0.48986519530485767],
                 'Vitoria-Gasteiz':[42.84966375015041, -2.6729523963530823],
                 'Gijón':[43.542387374632156, -5.662660018182284],
                 'Valladolid':[41.65351852763409, -4.72647259837896],
                 'Logroño':[42.46506702206999, -2.4457956229579314],
                 'Badajoz':[38.881447177614916, -6.970897576816443],
                 'La Coruña':[43.36561176102946, -8.407737490811884],
                 'Toledo':[39.859674039682886, -4.023829851005259],
                 'Castellón de la Plana':[39.985606076856385, -0.04109725786947045],
                 'Girona':[41.982870661094104, 2.822261199090652],
                 'Pamplona':[42.814339638624915, -1.6440447729253458],
                 'Coimbra':[40.20410565017061, -8.406776857540061],
                 'Pontevedra':[42.43223511518365, -8.644142729680269],
                 'Vigo':[42.233504963273916, -8.726699757450815],
                 'Cordoba':[37.8854226218, -4.77962961258667],
                 'Burgos':[42.347016677889904, -3.6901727471082246],                 
                 'Almería':[36.83924841252471, -2.464324847033624],
                 'Cartagena':[37.611685654929815, -0.9859738441731685],
                 'Salamanca':[40.96635430849213, -5.66362643683314],
                 'Tarragona':[41.1172283686032, 1.251150534818639],
                 'Lleida':[41.61914446073169, 0.6227081428586524],
                 'Sabadell':[41.5472273756068, 2.106718306400181],
                 'Vic':[41.931141240890014, 2.2565448528965204],
                 'Santander':[43.458392705452034, -3.8184104809476755],
                 'Albacete':[38.99299726280011, -1.8580046954707474],
                 'Ciudad Real':[38.98612448613222, -3.927311823256516],
                 'Cáceres':[39.47340370513005, -6.372180558189858],
                 'Orense':[42.339847596940366, -7.865032566020249],
                 'Cadix':[36.51169555103866, -6.274828218888884],
                 'Alcañiz':[41.05049869749971, -0.1296928749963408],
                 'Ceuta':[35.89085923549498, -5.325049804404845],
                 'Melilla':[35.287962547957946, -2.9386902329411004],
                 'Marseille':[43.295679317637514, 5.379449159402588],
                 
                 'Montpellier':[43.61569961016273, 3.8660460716876006],
                 'Toulouse':[43.60250698288325, 1.4444001370537687],
                 'Bordeaux':[44.84066126555645, -0.5736646509040485],
                 'Tanger':[35.76879107847087, -5.8255437961970316],
                 'Alger':[36.75432623335199, 3.0597614581665598],
                 'Oran':[35.70347908886616, -0.6377359988760769],
                 'Oujda':[34.679847265428876, -1.9083767412959642],
                 'Rabat':[34.00048205814858, -6.85942681601717],
                 'Fès':[34.032222604763035, -5.003061854995897],
                 'Porto'     :  [41.154259, -8.623670]}

def ComputeWindDirection(longitudinal_speeds, latitudinal_speeds):
    # load libraries
    import math
    import numpy as np
    # Calculate the wind direction in degrees
    directions = np.arctan2(latitudinal_speeds, longitudinal_speeds) * 180 / math.pi
    directions[directions < 0] += 360    
    # Convert directions to North-related directions
    north_directions = (90 - directions) % 360    
    return north_directions    
    
def ComputeAngleBetweenDirections(direction1, direction2):
    # load libraries
    import math
    import numpy as np
    # Compute the absolute difference between the directions
    abs_diff = abs(direction2 - direction1)
    # Take the smaller angle between the directions
    angle = min(abs_diff, 360 - abs_diff)
    return angle        

def CumMode(mode,x):
    import numpy as np
    result = np.full([len(x)],np.nan)
    for i in range(len(result)):
        if np.isfinite(x[i:]).any()==False: continue
        if mode=='mean': result[i] = np.nanmean(x[i:])
        if mode=='min' : result[i] = np.nanmin(x[i:])
        if mode=='max' : result[i] = np.nanmax(x[i:])
    return result


def GetBorders(modes=['NUTS2','NUTS3'], resolution_nuts = '10', country='ES'):
    import shapefile
    import geopandas as gpd
    import numpy as np
    shapes_dict = {}
    for mode in modes:
        print(mode)
        if mode.startswith('NUTS'):
            ilev_nuts = int(mode[4:])
            #resolution = '10'      # 60:coarse / 10:intermediate
            fn = '/esarchive/scratch/hpetetin/output/pyeslib_data/regions_shapefile/ref-nuts-2016-{}m/NUTS_RG_{}M_2016_4326_LEVL_{}/NUTS_RG_{}M_2016_4326_LEVL_{}.shp'.\
                format(resolution_nuts,resolution_nuts,ilev_nuts,resolution_nuts,ilev_nuts)
            sf = shapefile.Reader(fn)
            fields = [x[0] for x in sf.fields][1:]
            shps = [s.points for s in sf.shapes()]
            records = sf.records()
            shapes = sf.shapes()
            countries = np.array([records[i][2] for i in np.arange(len(records))])
            w = np.where(countries==country)[0]
            shapes_dict[mode] = list(np.array(shapes)[w])
        elif mode=='AQZ':
            fn = '/esarchive/scratch/hpetetin/DATA/AQZ/b2g.ZoneGeometries.shp'
            gdf = gpd.GeoDataFrame.from_file(fn)
            gdf = gdf[gdf['Country']==country]
            # Define coordinate reference systems (CRS)                                                                                      
            crs_4326 = pyproj.CRS.from_epsg(4326)   #(EPSG:4326, lon/lat coordinates in degrees)                                             
            # Convert to lon/lat (for information purpose)                                                                                   
            gdf_lonlat = gdf.to_crs(crs_4326)
            shapes_dict[mode] = list(gdf_lonlat['geometry'])
    return shapes_dict


def RunningMean(x,y,N,Nmin=None):    
    '''    
    Compute running mean
    ---------------------------------------------------
    '''
    import numpy as np
    if Nmin is None: Nmin=N-2
    nnew = int(len(x)-N+1)
    ynew = np.full([nnew], np.nan)
    yold = np.full([len(y)], np.nan) 
    for i in np.arange(nnew):
        w = np.arange(N)+i
        wok = np.where(np.isfinite(y[w])==True)[0]
        if len(wok) >= Nmin: 
            ynew[i] = np.mean(y[w][wok])
            yold[i+int(N/2)] = np.mean(y[w][wok])
    xnew = x[int(N/2):int(N/2)+nnew]
    
    return((xnew,ynew,yold))


def TestSignificance(x0,x1,verbose=True,test='Mann-Whitney U rank test'):
    from scipy import stats
    # test normality of x1 and x2 through Shapiro-Wilk test
    #    H0: normally distributed (if pvalue <= 0.05 => H0 can be rejected => not normally distributed
    #    H1 : not normallly distributed (if pvalue > 0.05 => H0 cannot be rejected => maybe normally distributed
    # (https://quantifyinghealth.com/report-shapiro-wilk-test/)
    if verbose:
        print('| Test normality of x through Shapiro-Wilk test')
        print('|    H0: normally distributed')
        print('|    H1: not normally distributed') 
    swtest = {}
    normaldistribution = True
    for ix,x in enumerate([x0,x1]): 
        swtest[str(ix)] = stats.shapiro(x)
        if swtest[str(ix)][1] <= 0.05:
            if verbose: print('| W={:6.2f},pvalue={:8.3f} <= 0.05 => H0 can be rejected => x{} not normally distributed'.\
                              format(swtest[str(ix)][0],swtest[str(ix)][1],ix))
            normaldistribution = False
        else:
            if verbose: print('| W={:6.2f},pvalue={:8.3f} >  0.05 => H0 cannot be rejected => x{} may be normally distributed'.\
                              format(swtest[str(ix)][0],swtest[str(ix)][1],ix))
            #print('not normal')

    normaldistribution = False

    if test=='ks_2samp':
        kstest = stats.ks_2samp(x0,x1)
        return (swtest['0'][0],swtest['0'][1],swtest['1'][0],swtest['1'][1],kstest.statistic,kstest.pvalue)     
    elif test=='brunnermunzel':
        brunnermunzeltest = stats.brunnermunzel(x0,x1)
        return (swtest['0'][0],swtest['0'][1],swtest['1'][0],swtest['1'][1],brunnermunzeltest.statistic,brunnermunzeltest.pvalue)
            
    if normaldistribution==False or test=='Mann-Whitney U rank test':
        if verbose:
            print('|   ==> the Shapiro-Wilk test shows that the distribution of at least one of the two variables ')
            print('|       departs significantly from normality, which prevents the use of the Welch\'s t-test')
            print('| Proceed with Mann-Whitney U rank test:')
            print('|   H0: x0 and x1 have a similar distribution')
            print('|   H1: x0 and x1 have a different distribution')
        mwuranktest = stats.mannwhitneyu(x0,x1)
        if mwuranktest.pvalue <= 0.05:
            if verbose: print('| U={:6.2f},pvalue={:8.3f} <= 0.05 => H0 can be rejected => u0≠u1'.format(mwuranktest.statistic,mwuranktest.pvalue))
        else:
            if verbose: print('| U={:6.2f},pvalue={:8.3f} >  0.05 => H0 cannot be rejected => may be u0=u1'.format(mwuranktest.statistic,mwuranktest.pvalue))
              
        return (swtest['0'][0],swtest['0'][1],swtest['1'][0],swtest['1'][1],mwuranktest.statistic,mwuranktest.pvalue)
    
    else:
        if verbose:
            print('|   ==> the Shapiro-Wilk test does not show evidence of non-normality for any of the two variables')
            print('|       so we can proceed with the Welch\'s t-test:')
            print('|   H0: u0=u1')
            print('|   H1: u0≠u1')
        ttest = stats.ttest_ind(x0,x1,equal_var=False)
        if ttest.pvalue <= 0.05:
            if verbose: print('| T={:6.2f},pvalue={:8.3f} <= 0.05 => H0 can be rejected => u0≠u1'.format(ttest.statistic,ttest.pvalue))
        else:
            if verbose: print('| T={:6.2f},pvalue={:8.3f} >  0.05 => H0 cannot be rejected => may be u0=u1'.format(ttest.statistic,ttest.pvalue))
        return (swtest['0'][0],swtest['0'][1],swtest['1'][0],swtest['1'][1],ttest.statistic,ttest.pvalue)



def NesToXarray(nesdata):
    import xarray as xr
    import numpy as np
    variables = list(nesdata.variables.keys())
    for ivar,var in enumerate(variables):
        # get data
        values = nesdata.variables[var]['data']                                                                                        
        tmp = xr.Dataset({var:(['time','lev','lat','lon'],values)},
                         coords={'time': np.array(nesdata.time),
                                 'lat' : np.array(nesdata.lat['data']),
                                 'lon' : np.array(nesdata.lon['data']),
                                 'lev' : np.array(nesdata.lev['data'])})
        # add variable attributes
        for x in ['units','long_name','standard_name']:
            try:
                tmp[var].attrs.update({x:nesdata.variables[var][x]})
            except:
                do_nothing = True
        # merge
        result = tmp if ivar==0 else result.merge(tmp)
    # add lon/lat attributes
    for x in ['units','long_name','standard_name','bounds']:
        try:
            result.lon.attrs.update({x:nesdata.lon[x]})
            result.lat.attrs.update({x:nesdata.lon[x]})
        except:
            do_nothing = True
    return result
            

def NesToPcolormeshFormat(ds_nes,grid='regular'):
    import numpy as np
    import xarray as xr
    # get dimensions
    ny,nx = ds_nes.dims['lat'],ds_nes.dims['lon']
    if grid=='regular':
        # expand lat and lon in 2D
        lat = np.tile(ds_nes.lat.values,(nx,1)).transpose()
        lon = np.tile(ds_nes.lon.values,(ny,1))
        # get the lat_b and lon_b first rows
        lat_b_0 = np.append(ds_nes.lat_bnds[:,0],ds_nes.lat_bnds[-1,-1])
        lon_b_0 = np.append(ds_nes.lon_bnds[:,0],ds_nes.lon_bnds[-1,-1])
        # expand lat_b and lon_b in 2D
        lat_b = np.tile(lat_b_0,(nx+1,1)).transpose()
        lon_b = np.tile(lon_b_0,(ny+1,1))
    # create xarray dataset
    ds = xr.Dataset({'lon'  :(['y','x'],lon),
                     'lat'  :(['y','x'],lat),
                     'lon_b':(['y_b','x_b'],lon_b),
                     'lat_b':(['y_b','x_b'],lat_b)})
    return ds




def PrepareMonarch(grid,htime,dtime,timescales_pollutants,exp,domain='regional_i01',nlevels=24,gridname=None):
    # load libraries
    import copy
    import numpy as np
    import os
    import pandas as pd
    import xarray as xr
    #from nes import *

    criteria = 0.75

    if exp=='a4g0': domain = 'd02'
    
    for itspol,tspol in enumerate(timescales_pollutants):
        print('{} ({}-{})'.format(tspol,htime[0].strftime('%Y%m%d%H'),htime[-1].strftime('%Y%m%d%H')))

        # get pollutant and timescale
        values = tspol.split('_')
        ts = values[0]
        pol = '_'.join('{}' for i in range(len(values)-1)).format(*values[1:])

        # get pollutant name corresponding to the file name
        polfn = copy.deepcopy(pol)
        obsmod = 'obs' if 'obs' in pol else 'mod'
        amf = 'AMFmonarch' if 'AMFmonarch' in pol else 'AMFtm5'       
        if pol.startswith('trcno2_pal') : polfn = 'trcno2_pal'
        if pol.startswith('trcno2_rpro') : polfn = 'trcno2_rpro'
        if pol.startswith('trcno2_rprooffl') : polfn = 'trcno2_rprooffl'
        if pol.startswith('trcno2_offl'): polfn = 'trcno2_offl'
        if pol.startswith('trcno2_comb'): polfn = 'trcno2_comb'
        if pol.startswith('trcform')    : polfn = 'trcform'
        if pol.startswith('trc')        : pol = '{}_{}_{}'.format(polfn,obsmod,amf)  
        
        # create empty dataset for hourly data
        ds_h = xr.Dataset({'h' : (['htime','y','x'], np.full([len(htime),grid.dims['y'],grid.dims['x']],np.nan))},
                          coords = {'htime': htime})
        
        # loop on daily time
        fn_last = None
        for itime,time in enumerate(dtime):

            # get file name            
            if pol.startswith('trc'): #(tropopsheric columns)
                fn = '/esarchive/scratch/hpetetin/output/cso/cso_formatted/{}/hourly/{}/{}_000_{}00.nc'.format(exp,polfn,polfn,pd.to_datetime(time).strftime('%Y%m%d'))

            elif pol in ['ws10','wd10']: #(wind)
                fn = '/esarchive/exp/monarch/{}/{}/hourly/{}/{}-000_{}00.nc'.format(exp,domain,'u10','u10',pd.to_datetime(time).strftime('%Y%m%d'))
                u10 = xr.open_dataset(fn)
                u10.close()
                fn_v10 = '/esarchive/exp/monarch/{}/{}/hourly/{}/{}-000_{}00.nc'.format(exp,domain,'v10','v10',pd.to_datetime(time).strftime('%Y%m%d'))
                v10 = xr.open_dataset(fn_v10)
                v10.close()
                if pol=='wd10':
                    tmp = u10.rename({'u10':'wd10'})
                    tmp.wd10[dict()] = ComputeWindDirection(tmp.wd10.values,v10.v10.values)
                    tmp.wd10.attrs['units'] = '°'
                elif pol=='ws10':
                    tmp = u10.rename({'u10':'ws10'})
                    tmp.ws10[dict()] = ((tmp.ws10.values)**2+(v10.v10.values)**2)**0.5
                    
            elif pol.startswith('aemis'): #(anthropogenic emissions)
                fn = '/esarchive/exp/monarch/{}/HERMESv3_OUT/HERMESv3_d01_{}00.nc'.format(exp,pd.to_datetime(time).strftime('%Y%m%d'))
                
            else: #(other)                
                fn = '/esarchive/exp/monarch/{}/{}/hourly/{}/{}-000_{}00.nc'.format(exp,domain,polfn,polfn,pd.to_datetime(time).strftime('%Y%m%d'))
                
            # skip if missing file or already read file
            if os.path.isfile(fn)==False or fn==fn_last:
                print('SKIP {}'.format(fn))
                continue
            print('READ {}'.format(fn))
            
            # read
            if not pol in ['ws10','wd10']:  
                tmp = xr.open_dataset(fn)
                tmp.close()                             
            fn_last = copy.deepcopy(fn)

            # rename lon/lat if necessary
            #dims = list(tmp.dims.keys())
            #print(dims)
            #if not 'lon' in dims and 'rlon' in dims: tmp = tmp.rename_dims({'rlon':'lon'})
            #if not 'lat' in dims and 'rlat' in dims: tmp = tmp.rename_dims({'rlat':'lat'})
            
            #zz
            
            # handle specific case of tropospheric columns
            #if pol.startswith('trc'):
            #    ts,pol,obsmod,amf = tspol.split('_')
            #    #tmp = tmp['{}_{}_{}'.format(pol,obsmod,amf)].rename('{}_{}'.format(pol,amf) ).to_dataset()
            #    pol = '{}_{}_{}'.format(pol,obsmod,amf)
            
            # special treatment eventually (compute total anthropogenic VOC and NOx emissions)
            if pol.startswith('aemis'):
                par_list = {'aemisVOC':['PAR', 'OLE', 'TOL', 'XYL', 'FORM', 'ALD2', 'ETH', 'ISOP', 'MEOH', 'ETOH',
                                        'ETHA', 'IOLE', 'ALDX', 'TERP', 'BENZENE', 'SESQ'],
                            'aemisNOx':['NO', 'NO2']}[pol]
                for ipar,par in enumerate(par_list):                
                    if ipar==0:
                        aemis_units = tmp[par].attrs['units']
                        tmp = tmp.rename({par:pol})
                    else:
                        tmp[pol][dict()] = tmp[pol].values + tmp[par].values
                        
            # keep only first level for concentrations, or sum levels for emissions
            if 'lev' in list(tmp.dims.keys()):
                if pol.startswith('aemis'):
                    tmp_rotated_pole = tmp[['rotated_pole']]
                    tmp = tmp[[pol]].sum('lev')
                else:
                    tmp = tmp.isel(lev=0)
                    
            # regrid if anthropogenic emissions (because currently saved on rotated lon-lat grid)
            if pol.startswith('aemis'):
                '''
                args = {}
                args['path_nes_grid'] = '/esarchive/scratch/hpetetin/output/mitigate/preprocessed_data/nes_grids'
                args['path_nes_weights'] = '/esarchive/scratch/hpetetin/output/mitigate/preprocessed_data/nes_weights'
                                
                # read grids
                fn_nes_grid = '{}/grid_{}_nes.nc'.format(args['path_nes_grid'],gridname)
                nes_grid = open_netcdf(path=fn_nes_grid, info=True)
                nes_grid.load()
                # read data with nes
                if os.path.isfile(fn)==False: continue
                nes_input = open_netcdf(path=fn)
                nes_input.load()
                # interpolate
                method = 'NearestNeighbour4'
                fn_weights = '{}/weights_HERMES{}_to_{}_{}.nc'.format(args['path_nes_weights'],exp,gridname,method)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if method.startswith('NearestNeighbour'):
                        n_neighbours = int(method.split('NearestNeighbour')[1])
                        nes_interp = nes_input.interpolate_horizontal(nes_grid,kind='NearestNeighbour',weight_matrix_path=fn_weights,n_neighbours=n_neighbours,info=False)
                    elif method=='Conservative':
                        nes_interp = nes_input.interpolate_horizontal(nes_grid, kind='Conservative',weight_matrix_path=fn_weights,info=False)
                # compute data for dataset
                ds_nes_interp = NesToXarray(nes_interp)

                print(ds_nes_interp)
                zz
                #.resample(time='1D').mean().isel(lev=0)
                wtime = np.intersect1d(ds.dtime.values,ds_nes_interp.time.values, assume_unique=False, return_indices=True)
                ds.d_data[dict(dtime=wtime[1])] = ds_nes_interp[var].values[wtime[2]]
                '''
                
                import xesmf as xe
                grid_aemis = GetLonLatGridFromReducedMONARCH(tmp.merge(tmp_rotated_pole))
                method = 'conservative'
                weightfile = '/esarchive/scratch/hpetetin/output/mitigate/regridding_weights/weights_rIP_0.1deg_to_{}_{}.nc'.format(gridname,method)
                regridder = xe.Regridder(grid_aemis, grid, method, reuse_weights=True, filename=weightfile)
                tmp = tmp.drop(['rlon','rlat']).rename({'rlon':'x','rlat':'y'}).merge(grid_aemis[['lon_b','lat_b']])
                tmp = regridder(tmp.drop(['lon_b','lat_b'])).merge(grid[['lon_b','lat_b']])
                
                
            # get unit convertion factor
            factor = 1
            if pol.startswith('trc'):                     # tropospheric columns (mol/m2 => Pmolec/cm2)
                factor = tmp[pol].attrs['multiplication_factor_to_convert_to_molecules_percm2']/1e15
                ds_h.h.attrs['units'] = 'Pmolec/cm2'
            elif pol.startswith('sconc') or pol=='nmvoc': # surface mixing ratios (nmol/mol (i.e. ppmv) => ppbv)
                factor = 1e3
                ds_h.h.attrs['units'] = 'ppbv'
            elif pol in ['pm10','pm2p5']:                 # surface concentrations (kg/m3 => ug/m3)
                factor = 1e9
                ds_h.h.attrs['units'] = 'ug/m3'
            elif pol.startswith('aemis'):
                factor = 1
                ds_h.h.attrs['units'] = aemis_units
            
            # convert unit
            #if pol.startswith('aemis'):
            #    tmp[pol.split('aemis')[1]][dict()] = tmp[pol.split('aemis')[1]].values * factor
            #else:
            tmp[pol][dict()] = tmp[pol].values * factor
            
            # transpose dimensions
            if pol.startswith('aemis')==False:
                tmp = tmp.transpose('time','lat','lon')
                
            # get data for 2D variables
            if pol.startswith('aemis') and 0:
                for xpol in aemis_variables: 
                    # get intersect times
                    wtime_intersect = np.intersect1d(ds_h.htime.values, tmp.time.values, return_indices=True)
                    if len(wtime_intersect[1])==0 or len(wtime_intersect[2])==0: continue
                    ds_h.h[dict(htime=wtime_intersect[1])] = tmp[xpol][dict(time=wtime_intersect[2])].values
            else:
                # get intersect times
                wtime_intersect = np.intersect1d(ds_h.htime.values, tmp.time.values, return_indices=True)
                if len(wtime_intersect[1])==0 or len(wtime_intersect[2])==0: continue
                ds_h.h[dict(htime=wtime_intersect[1])] = tmp[pol][dict(time=wtime_intersect[2])].values

        # adapt criteria if tropospheric columns
        if pol.startswith('trc'):
            criteria = 0.0
            
        # get desired timescales
        if ts=='h':
            # get hourly values
            ds = copy.deepcopy(ds_h.h)
            # rename and convert back to dataset  
            ds = ds.rename(tspol).to_dataset() 
            
        elif ts in ['d','d1max']:
            # get data availability mask (only 1 and NaN)
            count = ds_h.h.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute daimy maximum 8-hour average accounting for data availability
            if ts=='d'    : da = ds_h.h.resample(htime='1D').mean() * mask
            if ts=='d1max': da = ds_h.h.resample(htime='1D').max()  * mask
            # rename and convert back to dataset
            ds = da.rename(tspol).rename({'htime':'dtime'}).to_dataset()
            
        elif ts=='d8max':
            # compute rolling average over window
            window = 8
            da_rolling = ds_h.h.rolling(htime=window,min_periods=int(criteria*window),center=False).mean()
            # get data availability mask (only 1 and NaN)
            count = da_rolling.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)                                
            # compute daily maximum 8-hour average accounting for data availability
            da = (da_rolling.resample(htime='1D').max() * mask)
            # rename and convert back to dataset
            ds = da.rename(tspol).rename({'htime':'dtime'}).to_dataset()
            
        elif ts in ['mh']:
            # get data availability mask (only 1 and NaN)
            count = ds_h.h.resample(htime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from hourly values accounting for data availability
            da = (ds_h.h.resample(htime='1M').mean() * mask)
            # rename and convert back to dataset
            ds = da.rename(tspol).rename({'htime':'mtime'}).to_dataset() 
            
        elif ts in ['md']:            
            # get data availability mask (only 1 and NaN)
            count = ds_h.h.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # get daily average from hourly values accounting for data availability
            da_d = ds_h.h.resample(htime='1D').mean() * mask
            # rename and convert back to dataset
            ds_d = da_d.rename('d').rename({'htime':'dtime'}).to_dataset()  
            
            # get data availability mask (only 1 and NaN)
            count = ds_d.d.resample(dtime='1MS').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            da = ds_d.d.resample(dtime='1MS').mean() * mask
            # rename and convert back to dataset
            ds = da.rename(tspol).rename({'dtime':'mtime'}).to_dataset() 
            
        elif ts in ['md1max']:            
            # get data availability mask (only 1 and NaN)
            count = ds_h.h.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # get daily average from hourly values accounting for data availability
            da_d1max = ds_h.h.resample(htime='1D').max() * mask
            # rename and convert back to dataset  
            ds_d1max = da_d1max.rename('d1max').rename({'htime':'dtime'}).to_dataset()
            
            # get data availability mask (only 1 and NaN)
            count = ds_d1max.d1max.resample(dtime='1MS').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            da = ds_d1max.d1max.resample(dtime='1MS').mean() * mask
            # rename and convert back to dataset
            ds = da.rename(tspol).rename({'dtime':'mtime'}).to_dataset()  
                        
        elif ts in ['md8max']:
            # compute rolling average over window
            window = 8
            da_rolling = ds_h.h.rolling(htime=window,min_periods=int(criteria*window),center=False).mean()
            # get data availability mask (only 1 and NaN)
            count = da_rolling.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)                                
            # compute daily maximum 8-hour average accounting for data availability
            da_d8max = da_rolling.resample(htime='1D').max() * mask                  
            # rename and convert back to dataset  
            ds_d8max = da_d8max.rename('d8max').rename({'htime':'dtime'}).to_dataset()

            # get data availability mask (only 1 and NaN)
            count = ds_d8max.d8max.resample(dtime='1MS').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            da = ds_d8max.d8max.resample(dtime='1MS').mean() * mask
            # rename and convert back to dataset
            ds = da.rename(tspol).rename({'dtime':'mtime'}).to_dataset()
            
        # concatenate results
        result = copy.deepcopy(ds) if itspol==0 else result.merge(ds)

        # rename
        result = result.rename({tspol:'{}_{}'.format(tspol,exp)})
        MemoryUsage()
        
    #print('='*100)
    #print(result)
    #print('='*100)
    return result



def GetMonarchHourlyData(grid,htime,dtime,pollutants,exp,domain='regional_i01',nlevels=24,gridname=None):
    # load libraries
    import copy
    import numpy as np
    import os
    import pandas as pd
    import xarray as xr
    
    # create empty dataset
    ds= xr.Dataset({'h_sdata' : (['pollutant','htime','y','x'], np.full([len(pollutants),len(htime),grid.dims['y'],grid.dims['x']],np.nan))},
                   coords = {'pollutant': np.array(pollutants,dtype='object'),
                             'htime': htime})
            
    # loop on daily time
    fn_last = None
    for itime,time in enumerate(dtime):
        # loop on pollutants
        for ipol,pol in enumerate(pollutants):
            if pol in ['vertstrat']: continue
            
            # get pollutant name corresponding to the file name
            polfn = copy.deepcopy(pol)
            if '_mod' in pol: polfn = pol.split('_mod')[0]
            if '_obs' in pol: polfn = pol.split('_obs')[0]
            
            # get file name            
            if pol.startswith('trc'):
                fn = '/esarchive/scratch/hpetetin/output/cso/cso_formatted/{}/hourly/{}/{}_000_{}00.nc'.format(exp,polfn,polfn,pd.to_datetime(time).strftime('%Y%m%d'))
            elif pol.startswith('aemis'):
                fn = '/esarchive/exp/monarch/{}/HERMESv3_OUT/HERMESv3_d01_{}00.nc'.format(exp,pd.to_datetime(time).strftime('%Y%m%d'))
            else:
                fn = '/esarchive/exp/monarch/{}/{}/hourly/{}/{}-000_{}00.nc'.format(exp,domain,polfn,polfn,pd.to_datetime(time).strftime('%Y%m%d'))

            # skip if missing file
            if os.path.isfile(fn)==False:
                #print('{} missing'.format(fn))
                continue
            
            # read (if not already read)
            if fn!=fn_last:
                fn_last = copy.deepcopy(fn)
                tmp = xr.open_dataset(fn)
                tmp.close()

                # compute total anthropogenic VOC and NOx emissions
                if 'aemisVOC' in pollutants and pol.startswith('aemis'):
                    par_list = ['PAR', 'OLE', 'TOL', 'XYL', 'FORM', 'ALD2', 'ETH', 'ISOP', 'MEOH', 'ETOH',
                                'ETHA', 'IOLE', 'ALDX', 'TERP', 'BENZENE', 'SESQ']                    
                    for ipar,par in enumerate(par_list):
                        if ipar==0:
                            tmp = tmp.rename({par:'VOC'})
                        else:
                            tmp['VOC'][dict()] = tmp['VOC'].values + tmp[par].values
                if 'aemisNOx' in pollutants and pol.startswith('aemis'):
                    par_list = ['NO', 'NO2']
                    for ipar,par in enumerate(par_list):
                        if ipar==0:
                            tmp = tmp.rename({par:'NOx'})
                        else:
                            tmp['NOx'][dict()] = tmp['NOx'].values + tmp[par].values

                # keep only first level
                if 'lev' in list(tmp.dims.keys()):
                    if pol.startswith('aemis'):
                        aemis_variables = [i for i in list(tmp.keys()) if 'aemis{}'.format(i) in ds.pollutant.values]
                        tmp_rotated_pole = tmp[['rotated_pole']]
                        tmp = tmp[aemis_variables].sum('lev')
                    else:
                        tmp = tmp.isel(lev=0)
                
                # regrid if anthropogenic emissions (because saved on rotated lon-lat grid)
                if pol.startswith('aemis'):
                    import xesmf as xe
                    grid_aemis = GetLonLatGridFromReducedMONARCH(tmp.merge(tmp_rotated_pole))
                    method = 'conservative'
                    weightfile = '/esarchive/scratch/hpetetin/output/mitigate/regridding_weights/weights_rIP_0.1deg_to_{}_{}.nc'.format(gridname,method)
                    regridder = xe.Regridder(grid_aemis, grid, method, reuse_weights=True, filename=weightfile)
                    tmp = tmp.drop(['rlon','rlat']).rename({'rlon':'x','rlat':'y'}).merge(grid_aemis[['lon_b','lat_b']])
                    tmp = regridder(tmp.drop(['lon_b','lat_b'])).merge(grid[['lon_b','lat_b']])
                
            # convert units 
            factor = 1
            if pol.startswith('trc'):
                # tropospheric columns (mol/m2 => Pmolec/cm2)
                factor = tmp[pol].attrs['multiplication_factor_to_convert_to_molecules_percm2']/1e15
            elif pol.startswith('sconc') or pol=='nmvoc':
                # surface mixing ratios (nmol/mol (i.e. ppmv) => ppbv)
                factor = 1e3
            elif pol in ['pm10','pm2p5']:
                # surface concentrations (kg/m3 => ug/m3)
                factor = 1e9
            if pol.startswith('aemis'):
                tmp[pol.split('aemis')[1]][dict()] = tmp[pol.split('aemis')[1]].values * factor
            else:
                tmp[pol][dict()] = tmp[pol].values * factor
            
            # transpose dimensions
            if pol.startswith('aemis')==False:
                tmp = tmp.transpose('time','lat','lon')
                
            # get data for 2D variables
            if pol.startswith('aemis'):
                # loop on all useful internal pollutants included
                #pol_aemis_list = [i.split('aemis')[1] for i in ds.pollutant.values if i in list(tmp.keys())]
                #print(pol_aemis_list)
                
                for xpol in aemis_variables: #pol_aemis_list:
                    #print(xpol)
                    ipol_ds = np.where(ds.pollutant.values=='aemis{}'.format(xpol))[0][0]
                    # get intersect times
                    wtime_intersect = np.intersect1d(ds.htime.values, tmp.time.values, return_indices=True)
                    if len(wtime_intersect[1])==0 or len(wtime_intersect[2])==0: continue
                    ds.h_sdata[dict(pollutant=ipol_ds,htime=wtime_intersect[1])] = tmp[xpol][dict(time=wtime_intersect[2])].values
            else:
                # loop on all useful internal pollutants included
                for xpol in np.intersect1d(ds.pollutant.values, np.array(list(tmp.keys()))):
                    ipol_ds = np.where(ds.pollutant.values==xpol)[0][0]
                    # get intersect times
                    wtime_intersect = np.intersect1d(ds.htime.values, tmp.time.values, return_indices=True)
                    if len(wtime_intersect[1])==0 or len(wtime_intersect[2])==0: continue
                    ds.h_sdata[dict(pollutant=ipol_ds,htime=wtime_intersect[1])] = tmp[xpol][dict(time=wtime_intersect[2])].values
                
    return ds


def GetMonarchHourlyDataParallel(grid,htime,dtime,pollutants,exp,n_jobs=12,domain='regional_i01',nlevels=24,gridname=None): 
    # load libraries
    import copy
    import numpy as np
    import os
    import pandas as pd
    import xarray as xr
    import multiprocessing

    # create empty dataset
    ds= xr.Dataset({'h_sdata' : (['pollutant','htime','y','x'], np.full([len(pollutants),len(htime),grid.dims['y'],grid.dims['x']],np.nan))},
                   coords = {'pollutant': np.array(pollutants,dtype='object'),
                             'htime': htime})

    # get groups
    grps_dtime = SplitArray(dtime,n_groups=n_jobs)
    grps_htime = []
    for grp in grps_dtime:
        grp_htime = []
        for i in grp:        
            grp_htime += list(i*24+np.arange(24))            
        grps_htime.append(grp_htime)
        
    # get hourly dataset
    if n_jobs > 1:
        jobargs = [(grid,htime[grps_htime[igrp]],dtime[grps_dtime[igrp]],pollutants,exp,domain,nlevels,gridname) for igrp in range(len(grps_dtime))]
        pool = multiprocessing.Pool(n_jobs)
        print('Launching GetMonarchHourlyData in parallel (#={})...'.format(len(jobargs)))
        allres = pool.starmap(GetMonarchHourlyData, jobargs)
        pool.close()
        pool.join()
        ds_hourly = None
        for igrp,res in enumerate(allres):
            ds_hourly = copy.deepcopy(res) if ds_hourly is None else xr.concat([ds_hourly,res],dim='htime')
    else:
        ds_hourly = None
        print('Launching GetMonarchHourlyData in sequential (#={})...'.format(len(grps_dtime)))
        for igrp in range(len(grps_dtime)):
            print(igrp)
            res =  GetMonarchHourlyData(grid,htime[grps_htime[igrp]],dtime[grps_dtime[igrp]],pollutants,exp,domain,nlevels,gridname)
            ds_hourly = copy.deepcopy(res) if ds_hourly is None else xr.concat([ds_hourly,res],dim='htime')

    return ds_hourly


def GetMonarchDataParallel(grid,htime,dtime,mtime,pollutants,exp,timescales,n_jobs=12,
                           domain='regional_i01',nlevels=24,gridname=None,criteria=0.75):
    # load libraries
    import copy
    import numpy as np
    import os
    import pandas as pd
    import xarray as xr
    import multiprocessing

    # create empty dataset
    ds = None
    for its,ts in enumerate(timescales):
        if 1:
            if ts=='h':
                tmp = xr.Dataset({'h_sdata' : (['pollutant','htime','y','x'], np.full([len(pollutants),len(htime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'htime': htime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts=='d':
                tmp = xr.Dataset({'d_sdata' : (['pollutant','dtime','y','x'], np.full([len(pollutants),len(dtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'dtime': dtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts=='d1max':
                tmp = xr.Dataset({'d1max_sdata' : (['pollutant','dtime','y','x'], np.full([len(pollutants),len(dtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'dtime': dtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts=='d8max':
                tmp = xr.Dataset({'d8max_sdata' : (['pollutant','dtime','y','x'], np.full([len(pollutants),len(dtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'dtime': dtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts.startswith('m'):
                tmp = xr.Dataset({'m_sdata' : (['pollutant','mtime','y','x'], np.full([len(pollutants),len(mtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'mtime': mtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)

    # get hourly data
    print('get ds_h')
    ds_h = GetMonarchHourlyDataParallel(grid,htime,dtime,pollutants,exp,n_jobs,domain,nlevels,gridname=gridname)
        
    # loop on timescales
    for its,ts in enumerate(timescales):
        print(ts)
        if ts=='h':
            # get hourly values
            ds['h_sdata'][dict()] = ds_h['h_sdata'].values

        elif ts in ['d','d1max']:
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute daimy maximum 8-hour average accounting for data availability
            if ts=='d'    : ds['d_sdata'][dict()]     = (ds_h['h_sdata'].resample(htime='1D').mean() * mask).values
            if ts=='d1max': ds['d1max_sdata'][dict()] = (ds_h['h_sdata'].resample(htime='1D').max()  * mask).values
            
        elif ts=='d8max':
            # compute rolling average over window
            window = 8
            da_rolling = ds_h['h_sdata'].rolling(htime=window,min_periods=int(criteria*window),center=False).mean()
            # get data availability mask (only 1 and NaN)
            count = da_rolling.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)                                
            # compute daily maximum 8-hour average accounting for data availability
            ds['d8max_sdata'][dict()] = (da_rolling.resample(htime='1D').max() * mask).values
            
        elif ts in ['m|h']:
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from hourly values accounting for data availability
            ds['m_sdata'][dict()]     = (ds_h['h_sdata'].resample(htime='1M').mean() * mask).values
            
        elif ts in ['m|d']:            
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # get daily average from hourly values accounting for data availability
            ds_d = (ds_h[['h_sdata']].resample(htime='1D').mean() * mask)
            # rename
            ds_d = ds_d.rename({'h_sdata':'d_sdata','htime':'dtime'}).transpose('pollutant','dtime','y','x')
            # get data availability mask (only 1 and NaN)
            count = ds_d['d_sdata'].resample(dtime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            ds['m_sdata'][dict()] = (ds_d['d_sdata'].resample(dtime='1M').mean() * mask).values
            # free memory
            del ds_d
            
        elif ts in ['m|d1max']:            
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # get daily average from hourly values accounting for data availability
            ds_d1max = (ds_h[['h_sdata']].resample(htime='1D').max() * mask)
            # rename
            ds_d1max = ds_d1max.rename({'h_sdata':'d_sdata','htime':'dtime'}).transpose('pollutant','dtime','y','x')
            # get data availability mask (only 1 and NaN)
            count = ds_d1max['d_sdata'].resample(dtime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            ds['m_sdata'][dict()] = (ds_d1max['d_sdata'].resample(dtime='1M').mean() * mask).values
            # free memory
            del ds_d1max
            
        elif ts in ['m|d8max']:
            # compute rolling average over window
            window = 8
            da_rolling = ds_h['h_sdata'].rolling(htime=window,min_periods=int(criteria*window),center=False).mean()
            # get data availability mask (only 1 and NaN)
            count = da_rolling.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)                                
            # compute daily maximum 8-hour average accounting for data availability
            ds_d8max = (da_rolling.resample(htime='1D').max() * mask)
            # rename
            ds_d8max = ds_d8max.rename({'htime':'dtime'}).transpose('pollutant','dtime','y','x')
            # get data availability mask (only 1 and NaN)
            count = ds_d8max.resample(dtime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            ds['m_sdata'][dict()] = (ds_d8max.resample(dtime='1M').mean() * mask).values
            # free memory
            del ds_d8max
            
    return ds
                        


def ComputeTimeScaleData(ds_h,grid,htime,dtime,mtime,timescales,criteria=0.75):
    # load libraries
    import copy
    import numpy as np
    import os
    import pandas as pd
    import xarray as xr
    import multiprocessing

    # get pollutants
    pollutants = ds_h.pollutant.values
    
    # create empty dataset
    ds = None
    for its,ts in enumerate(timescales):
        if 1: 
            if ts=='h':
                tmp = xr.Dataset({'h_sdata' : (['pollutant','htime','y','x'], np.full([len(pollutants),len(htime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'htime': htime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts=='d':
                tmp = xr.Dataset({'d_sdata' : (['pollutant','dtime','y','x'], np.full([len(pollutants),len(dtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'dtime': dtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts=='d1max':
                tmp = xr.Dataset({'d1max_sdata' : (['pollutant','dtime','y','x'], np.full([len(pollutants),len(dtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'dtime': dtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts=='d8max':
                tmp = xr.Dataset({'d8max_sdata' : (['pollutant','dtime','y','x'], np.full([len(pollutants),len(dtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'dtime': dtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
            elif ts.startswith('m'):
                tmp = xr.Dataset({'m_sdata' : (['pollutant','mtime','y','x'], np.full([len(pollutants),len(mtime),grid.dims['y'],grid.dims['x']],np.nan))},
                                 coords = {'pollutant': np.array(pollutants,dtype='object'),
                                           'mtime': mtime})
                ds = copy.deepcopy(tmp) if ds is None else ds.merge(tmp)
    
    # loop on timescales
    for its,ts in enumerate(timescales):
        if ts=='h':
            # get hourly values
            ds['h_sdata'][dict()] = ds_h['h_sdata'].values

        elif ts in ['d','d1max']:
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute daimy maximum 8-hour average accounting for data availability
            if ts=='d'    : ds['d_sdata'][dict()]     = (ds_h['h_sdata'].resample(htime='1D').mean() * mask).values
            if ts=='d1max': ds['d1max_sdata'][dict()] = (ds_h['h_sdata'].resample(htime='1D').max()  * mask).values
            
        elif ts=='d8max':
            # compute rolling average over window
            window = 8
            min_periods = int(criteria*window) if criteria!=0 else 1
            da_rolling = ds_h['h_sdata'].rolling(htime=window,min_periods=min_periods,center=False).mean()
            # get data availability mask (only 1 and NaN)
            count = da_rolling.resample(htime='1D').count()
            #mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            mask = count.where(count > int(criteria*24)).where(count <= int(criteria*24), 1.)
            
            # compute daily maximum 8-hour average accounting for data availability
            ds['d8max_sdata'][dict()] = (da_rolling.resample(htime='1D').max() * mask).values
            
        elif ts in ['m|h']:
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from hourly values accounting for data availability
            ds['m_sdata'][dict()]     = (ds_h['h_sdata'].resample(htime='1M').mean() * mask).values
            
        elif ts in ['m|d']:            
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # get daily average from hourly values accounting for data availability
            ds_d = (ds_h[['h_sdata']].resample(htime='1D').mean() * mask)
            # rename
            ds_d = ds_d.rename({'h_sdata':'d_sdata','htime':'dtime'}).transpose('pollutant','dtime','y','x')
            # get data availability mask (only 1 and NaN)
            count = ds_d['d_sdata'].resample(dtime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            ds['m_sdata'][dict()] = (ds_d['d_sdata'].resample(dtime='1M').mean() * mask).values
            # free memory
            del ds_d
            
        elif ts in ['m|d1max']:            
            # get data availability mask (only 1 and NaN)
            count = ds_h['h_sdata'].resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # get daily average from hourly values accounting for data availability
            ds_d1max = (ds_h[['h_sdata']].resample(htime='1D').max() * mask)
            # rename
            ds_d1max = ds_d1max.rename({'h_sdata':'d_sdata','htime':'dtime'}).transpose('pollutant','dtime','y','x')
            # get data availability mask (only 1 and NaN)
            count = ds_d1max['d_sdata'].resample(dtime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            ds['m_sdata'][dict()] = (ds_d1max['d_sdata'].resample(dtime='1M').mean() * mask).values
            # free memory
            del ds_d1max
            
        elif ts in ['m|d8max']:
            # compute rolling average over window
            window = 8
            min_periods = int(criteria*window) if criteria!=0 else 1
            da_rolling = ds_h['h_sdata'].rolling(htime=window,min_periods=min_periods,center=False).mean()
            # get data availability mask (only 1 and NaN)
            count = da_rolling.resample(htime='1D').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)                                
            # compute daily maximum 8-hour average accounting for data availability
            ds_d8max = (da_rolling.resample(htime='1D').max() * mask)
            # rename
            ds_d8max = ds_d8max.rename({'htime':'dtime'}).transpose('pollutant','dtime','y','x')
            # get data availability mask (only 1 and NaN)
            count = ds_d8max.resample(dtime='1M').count()
            mask = count.where(count >= int(criteria*24)).where(count < int(criteria*24), 1.)
            # compute monthly average from daily values accounting for data availability
            ds['m_sdata'][dict()] = (ds_d8max.resample(dtime='1M').mean() * mask).values
            # free memory
            del ds_d8max
            
    return ds
                        
    

    
def InvestMechanism(species_list,fn=None,nmax=1000):
    # load libraries
    import json
    import numpy as np

    # get file
    if fn is None: fn = '{}/chemical_mechanism/cb05_mechanism.json'.format(pyeslib_data_dir)

    # read json file
    f = open(fn)
    x = json.load(f)

    # get reactions data
    x = x['camp-data'][0]['reactions']
    nreactions = len(x)

    for spec in species_list:
        nreactants,nproducts = 0,0
        reactions = []
        reactions_reactants,reactions_products = [],[]
        type_reactants,type_products = [],[]
        
        for ireac in range(nreactions):
            reac = x[ireac]
            present = False
            if spec in list(reac['reactants'].keys()):
                nreactants += 1
                type_reactants.append(reac['type'])
                present = True
            elif spec in list(reac['products'].keys()):
                nproducts += 1
                type_products.append(reac['type'])
                present = True
            if present:
                tmp = list(reac['reactants'].keys())
                for ireactant,reactant in enumerate(tmp):
                    if 'qty' in list(reac['reactants'][reactant].keys()):
                        tmp[ireactant]  = '{}*{}'.format(reac['reactants'][reactant]['qty'], reactant)
                reaction_left_side = ' + '.join('{}' for i in tmp).format(*tmp)
            
                tmp_name = np.array(list(reac['products'].keys()))
                tmp_yield = []
                for prod in tmp_name:
                    if 'yield' in reac['products'][prod]:
                        tmp_yield.append('{}*'.format(reac['products'][prod]['yield']))
                    else:
                        tmp_yield.append('')
                if spec in list(reac['reactants'].keys()):
                    worder = np.arange(len(tmp_name)) 
                else:
                    worder = np.append(np.where(np.array(tmp_name)==spec)[0][0],np.where(np.array(tmp_name)!=spec)[0])
                tmp = ['{}{}'.format(tmp_yield[i],tmp_name[i]) for i in worder] 
                reaction_right_side = ' + '.join('{}' for i in tmp).format(*tmp)
                reaction_right_side = reaction_right_side.replace('+ -','- ')
                reaction = '({:15s}) {:16s} -> {}'.format(reac['type'],reaction_left_side,reaction_right_side)
            
                reactions.append(reaction)
                if spec in list(reac['reactants'].keys()):
                    reactions_reactants.append(reaction)             
                elif spec in list(reac['products'].keys()): 
                    reactions_products.append(reaction)   

        type_reactants,type_products = np.array(type_reactants),np.array(type_products)
        print('\n=================================================================')
        print('============================= {}'.format(spec))
        print('N(reactions as reactant) = {:5d}'.format(nreactants))
        for typ in np.unique(type_reactants):
            print('    {:15s} ({})'.format(typ,len(np.where(type_reactants==typ)[0])))
        for reac in reactions_reactants:
            if len(reac)>nmax:
                print('{:20s} | {} [...]'.format('',reac[:nmax]))
            else:
                print('{:20s} | {}'.format('',reac))
        print('N(reactions as product) = {:5d}'.format(nproducts))
        for typ in np.unique(type_products):
            print('    {:15s} ({})'.format(typ,len(np.where(type_products==typ)[0])))
        for reac in reactions_products:
            if len(reac)>nmax:
                print('{:20s} | {} [...]'.format('',reac[:nmax]))
            else:
                print('{:20s} | {}'.format('',reac))

                
def InvestMeganSpeciation(mechanismversion_list, count_terp_sesq=False, nmax=1000):
    # load libraries
    import numpy as np
    import pandas as pd
    
    # list of MEGAN terpenes and sesquiterpenes as seen from saprcnov  
    megan_terpenes = np.array(['pinene_a', 'pinene_b', 'myrcene', 'ocimene_al', 'ocimene_c_b',
                               'ocimene_t_b', 'camphene', 'bornene', 'fenchene_a', 'carene_3',
                               'fenchene_b', 'phellandrene_a', 'terpinene_g', 'terpinene_a',
                               'limonene', 'phellandrene_b', 'terpinolene', 'thujene_a',
                               'sabinene', 'verbenene', 'cymene_p', 'cymene_o', 'meta-cymenene',
                               'p-cymenene', 'piperitone', 'terpineol_a', 'estragole',
                               'terpineol_4', 'myrtenal', 'ionone_b', 'ipsenol',
                               'geranyl_acetone', 'benzene', 'linalool', 'salicylaldehyde',
                               'guaiacol', 'met_benzoate', 'safrole', 'eugenol', 'acetophenone',
                               'benzyl_alcohol', 'benzaldehyde', 'anisole', 'ethyl_cinnamate',
                               'cinnamaldehyde', 'toluene', 'phenol', 'met_salicylate',
                               'phenylacetaldehyde', 'cinnamic_acid', 'benzyl_acetate',
                               'coniferyl_alcohol', 'jasmone', 'chavicol', 'p_coumaric_acid',
                               '3_metthiophene', '3_metfuran', 'met_jasmonate', 'xylene',
                               'linalool_OXD_c', 'z3_hexen_1yl_butyrate', 'linalool_OXD_t',
                               '2met_nonatriene', 'terpinyl_ACT_a', 'skatole', 'naphthalene',
                               'indole', 'santene'])
    megan_sesquiterpenes = np.array(['caryophyllene_b', 'caryophyllene_c_i', 'cadinene_d', 'selinene_d',
                                     'bisabolene_b', 'farnescene_a', 'patchoulene_b', 'elemene_b',
                                     'nerolidol_c', 'humulene_a', 'muurolene_a', 'bergamotene_a',
                                     'germacrene_B', 'selinene_b', 'calarene', 'farnescene_b',
                                     'acoradiene', 'humulene_g', 'muurolene_g', 'germacrene_D',
                                     'cadinene_g', 'nerolidol_t', 'bergamotene_b', 'bisabolene_a',
                                     'homosalate', '2ethyl_hexyl_salate', 'cedrene_a', 'thujopsene',
                                     'longifolene', 'zingiberene_a', 'isolongifolene', 'longicyclene',
                                     'copaene_a', 'bourbonene_b', 'longipinene', 'cubebene_b',
                                     'ylangene_a', 'cubebene_a', 'copaene_b', 'kaur_16_ene',
                                     'gurjunene_b', 'aromadendrene', 'benzyl_benzoate', '8_heptadecene',
                                     '3met_3DCTT'])
    
    # loop on mechanisms
    for mechanismversion in mechanismversion_list:
        print('\n=================== {} ==================='.format(mechanismversion))

        mechanism,source = mechanismversion.split('|')[0],mechanismversion.split('|')[1]

        if source=='EXT':
            version = mechanismversion.split('|')[2]
            
            # get directory
            if version=='v2.1':
                include_dir = '{}/speciation_megan/INCLDIR_megan2.1'.format(pyeslib_data_dir)
            elif version=='v3.1':
                include_dir = '{}/speciation_megan/INCLDIR_megan3.1'.format(pyeslib_data_dir) 
        
            # ext file
            fn = '{}/MAP_CV2{}.EXT'.format(include_dir,mechanism)
            print('| Source: {}'.format(fn))
            
            # read Fortran file
            lines = open(fn, 'r').readlines()
            inside = False
            
            # initialize
            megan_species = []
            mechanism_species = []
            
            # loop on lines
            iorder=0
            for iline,line in enumerate(lines):
                line = line.replace(' ','').replace('\t','')
                if line[:1]=='&' and iorder==3:
                    iorder = 0
                if line[:1]=='&' and iorder==2:
                    iorder += 1
                if line[:1]=='&' and iorder==1:
                    mechanism_species.append(line.split("'")[1])
                    iorder +=1            
                if line[:1]=='&' and iorder==0:
                    megan_species.append(line.split("'")[1])
                    iorder +=1
                
            # convert to array
            megan_species = np.array(megan_species)
            mechanism_species = np.array(mechanism_species)
            
            # summary
            print('| Gas-hase chemical mechanism : {}'.format(mechanism))
            print('| Number of associations = {}'.format(len(megan_species)))
            print('| Number of (unique) MEGAN species = {}'.format(len(np.unique(megan_species))))
            print('| Number of (unique) mechanism species = {}'.format(len(np.unique(mechanism_species))))
            spec_list = list(np.unique(mechanism_species)) ; spec_list.sort()
            for ispec,spec in enumerate(spec_list):
                w = np.where(mechanism_species==spec)[0]
                tmp = megan_species[w]
                megan_species_selected = ';'.join('{}' for i in tmp).format(*tmp)

                if count_terp_sesq==True:
                    n_terpenes = len(np.intersect1d(tmp,megan_terpenes))
                    n_sesquiterpenes = len(np.intersect1d(tmp,megan_sesquiterpenes)) 
                    suffix = '[terp:{}|sesq:{}]'.format(n_terpenes,n_sesquiterpenes)
                else:
                    suffix = ''
                
                if len(megan_species_selected) > nmax: megan_species_selected = '{} [...]'.format(megan_species_selected[:nmax])
                print('|    {:10s} <= {:3d} MEGAN species: {} {}'.format(spec,len(w),megan_species_selected,suffix))

                
        elif source in ['FORTRAN-wrfchem','FORTRAN-nmmb-monarch','FORTRAN-auto-monarch','FORTRAN-auto-caliope']:

            # fortran file
            fn = '{}/speciation_megan/wrf_chem_module_data_mgn2mech.F'.format(pyeslib_data_dir)

            # read Fortran file
            lines = open(fn, 'r').readlines()
            inside = False
            
            if source=='FORTRAN-nmmb-monarch':
                print('| Source: module_data_mgn2mech.F90 (from nmmb-MONARCH)')
                subroutine = 'get_megan2{}_table_nmmbmonarch'.format(mechanism)
            elif source=='FORTRAN-auto-monarch':
                print('| Source: module_data_mgn2mech.F90 (from auto-MONARCH)')
                subroutine = 'get_megan2{}_table_automonarch'.format(mechanism)
            elif source=='FORTRAN-auto-caliope':
                print('| Source: module_data_mgn2mech.F90.likeauto-caliope (from auto-CALIOPE)')
                subroutine = 'get_megan2{}_table_autocaliope'.format(mechanism)
            elif source=='FORTRAN-wrfchem':
                print('| Source: wrf_chem_module_data_mgn2mech.F (from WRF)')
                subroutine = 'get_megan2{}_table'.format(mechanism)
                       
            
            # initialize
            megan_species = []
            mechanism_species = []
            inside = False 
            # loop on lines
            for iline,line in enumerate(lines):
                # modify the line
                line = line.replace(' ','').replace('\t','')
                # skip comment lines
                if line[:1]=='!' or line in ['','\n']: continue
                # check if outside
                if line.startswith('ENDSUBROUTINE{}'.format(subroutine)): inside = False
                # analyze
                if inside:
                    megan_species.append(line.split('=')[1].split(';')[0])
                    mechanism_species.append(line.split('=')[2].split(';')[0])
                # check if inside
                if line.startswith('SUBROUTINE{}'.format(subroutine)): inside = True

            # convert to array
            megan_species = np.array(megan_species)
            mechanism_species = np.array(mechanism_species)
        
            # summary
            print('| Gas-hase chemical mechanism : {}'.format(mechanism))
            print('| Number of associations = {}'.format(len(megan_species)))
            print('| Number of MEGAN species = {}'.format(len(np.unique(megan_species))))
            print('| Number of mechanism species = {}'.format(len(np.unique(mechanism_species))))
            spec_list = list(np.unique(mechanism_species)) ; spec_list.sort()

            for spec in spec_list:
                w = np.where(mechanism_species==spec)[0]
                tmp = megan_species[w]
                megan_species_selected = ';'.join('{}' for i in tmp).format(*tmp)

                if count_terp_sesq==True: 
                    n_terpenes = len(np.intersect1d(tmp,megan_terpenes))
                    n_sesquiterpenes = len(np.intersect1d(tmp,megan_sesquiterpenes)) 
                    suffix = '[terp:{}|sesq:{}]'.format(n_terpenes,n_sesquiterpenes)
                else:
                    suffix = ''
                    
                if len(megan_species_selected) > nmax: megan_species_selected = '{} [...]'.format(megan_species_selected[:nmax])
                print('|    {:10s} <= {:3d} MEGAN species: {} {}'.format(spec,len(w),megan_species_selected,suffix))

        elif source in ['HERMES']:
            
            # fortran file
            fn = '{}/speciation_megan/HERMESv2_Speciation_snap11.csv'.format(pyeslib_data_dir)

            # read file
            tab = pd.read_csv(fn,sep=';')

            # reshape species names
            tab['Specie'] = [i.replace("'","") for i in tab['Specie'].values]

            mechanism_species = np.array(list(tab.columns[1:]))
            megan_species = tab['Specie'].values
            number_associations = len(np.where(tab.values[:,1:].flatten()!=0)[0])

            print('| Gas-hase chemical mechanism : {}'.format(mechanism))
            print('| Number of associations = {}'.format(number_associations))
            print('| Number of MEGAN species = {}'.format(len(np.unique(megan_species))))
            print('| Number of mechanism species = {}'.format(len(np.unique(mechanism_species))))
            
            for spec in mechanism_species:
                w = np.where(tab[spec].values!=0)[0]
                tmp = megan_species[w]
                megan_species_selected = ';'.join('{}' for i in tmp).format(*tmp)      
                if count_terp_sesq==True: 
                    n_terpenes = len(np.intersect1d(megan_species_selected,megan_terpenes))
                    n_sesquiterpenes = len(np.intersect1d(megan_species_selected,megan_sesquiterpenes)) 
                    suffix = '[terp:{}|sesq:{}]'.format(n_terpenes,n_sesquiterpenes)
                else:
                    suffix = ''
                print('|    {:10s} <= {:3d} MEGAN species: {} {}'.format(spec,len(w),megan_species_selected,suffix))                 
            

            
def Summary(x):
    import numpy as np
    x = x.flatten()
    n,ntot = len(np.where(np.isfinite(x))[0]),len(x)
    print('mean={:0.2f} std={:0.2f} N={}/{} ({:0.1f}%)'.format(np.nanmean(x),np.nanstd(x),n,ntot,n/ntot*100))
    print('min={:0.2f} max={:0.2f}'.format(np.nanmin(x),np.nanmax(x)))
    print('p1/5/25={:0.2f}/{:0.2f}/{:0.2f} p50={:0.2f} p75/p95/p99={:0.2f}/{:0.2f}/{:0.2f}'.\
          format(np.nanpercentile(x,1),
                 np.nanpercentile(x,5),
                 np.nanpercentile(x,25),
                 np.nanpercentile(x,50),
                 np.nanpercentile(x,75),
                 np.nanpercentile(x,95),
                 np.nanpercentile(x,99)))

def SeparateOnSeveralLines(sentence,nmin=3,sep=' '):
    """ Split a character chain over several lines

    Parameters 
    ----------       
    sentence : str
        Chain of character to be splitted over several lines

    nmin : int
        Minimum number of characters to be put on a new line

    sep : str
        Separator

    Outputs
    -------

    output : str
        Chain of character splitted over several lines 

    Examples
    --------
    >>> from pyeslib import pyeslib
    >>> print(pyeslib.SeparateOnSeveralLines('Communidad de Madrid'))
    Communidad
    de Madrid 
    >>> print(pyeslib.SeparateOnSeveralLines('Communidad de Madrid',nmin=2))   
    Communidad
    de
    Madrid 
    """
    words = sentence.split(sep)
    nwords = len(words)
    result = ''
    for iword,word in enumerate(words):
        if iword < nwords-1:
            result = '{}{}\n'.format(result,word) if len(word) >= nmin else '{}{} '.format(result,word)
        else:
            result = '{}{} '.format(result,word)
    return result


def AddArea(grid,n_jobs=12):
    """ Compute and add cells area (in km2) into grid dataset

    Parameters 
    ----------       
    grid : xarray dataset
        Grid dataset

    n_jobs : int
        Number of cores available for parallelization

    Outputs
    -------

    output : xarray dataset
        Grid dataset with new variable giving the area of the different cells

    Examples
    --------
    >>> from pyeslib import pyeslib
    >>> extent,reslon,reslat = [-10., 5., 35., 45.], 1, 1
    >>> grid = pyeslib.DefineLonLatRegularGrid(extent, reslon, reslat)
    >>> grid = pyeslib.AddArea(grid)
    >>> print(grid)
    <xarray.Dataset>
    Dimensions:  (y: 10, x: 15, y_b: 11, x_b: 16)
    Dimensions without coordinates: y, x, y_b, x_b
    Data variables:
        lon      (y, x) float64 -9.5 -8.5 -7.5 -6.5 -5.5 ... 0.5 1.5 2.5 3.5 4.5
        lat      (y, x) float64 35.5 35.5 35.5 35.5 35.5 ... 44.5 44.5 44.5 44.5
        lon_b    (y_b, x_b) float64 -10.0 -9.0 -8.0 -7.0 -6.0 ... 2.0 3.0 4.0 5.0
        lat_b    (y_b, x_b) float64 35.0 35.0 35.0 35.0 35.0 ... 45.0 45.0 45.0 45.0
        area     (y, x) float64 1.007e+04 1.007e+04 ... 8.837e+03 8.837e+03
    Attributes:
        grid_type:              longitude-latitude regular grid
        extent:                 [-10.0, 5.0, 35.0, 45.0]
        resolution_degree_lon:  1
        resolution_degree_lat:  1
        creation_date:          2022-09-09 10:16:26.982378
    """
    # load libraries
    import multiprocessing
    import numpy as np
    import xarray as xr
    
    area_grid = np.full([grid.dims['y'],grid.dims['x']],np.nan)
    if n_jobs > 1:
        jobargs = [(grid,ix) for ix in range(grid.dims['x'])]
        pool = multiprocessing.Pool(n_jobs)
        allres = pool.starmap(ComputeAreaAlongIy, jobargs)
        pool.close()
        pool.join()
        for ix,res in enumerate(allres):
            area_grid[:,ix] = res
    else:
        for ix in range(grid.dims['x']):
            area_grid[:,ix] = compute_area_along_iy(grid,ix)
    grid = grid.merge(xr.Dataset({'area':(['y','x'],area_grid)}))
    return(grid)

    
def ComputeAreaAlongIy(grid,ix):
    """ Function to compute area at a given longitudinal band "ix"
    """
    # load libraries
    import shapely.geometry
    import pyproj
    import geopandas as gpd
    import pandas as pd
    
    # define coordinate reference systems (CRS)
    crs_4326 = pyproj.CRS.from_epsg(4326)   #(EPSG:4326, lon/lat coordinates in degrees)
    crs_3035 = pyproj.CRS.from_epsg(3035)   #(EPSG:3035, x/y coordinates in meters)  

    # prepare geopandas dataframe of cells
    list_polygon_cell_lonlat = []
    for iy in range(grid.dims['y']):
        lon_cell = [float(grid.lon_b.sel(x_b=ix  , y_b=iy  )),
                    float(grid.lon_b.sel(x_b=ix  , y_b=iy+1)),
                    float(grid.lon_b.sel(x_b=ix+1, y_b=iy+1)),
                    float(grid.lon_b.sel(x_b=ix+1, y_b=iy  )),
                    float(grid.lon_b.sel(x_b=ix  , y_b=iy  ))]
        lat_cell = [float(grid.lat_b.sel(x_b=ix  , y_b=iy  )),
                    float(grid.lat_b.sel(x_b=ix  , y_b=iy+1)),
                    float(grid.lat_b.sel(x_b=ix+1, y_b=iy+1)),
                    float(grid.lat_b.sel(x_b=ix+1, y_b=iy  )),
                    float(grid.lat_b.sel(x_b=ix  , y_b=iy  ))]
        polygon_cell_lonlat = shapely.geometry.Polygon(zip(lon_cell, lat_cell))
        list_polygon_cell_lonlat.append(polygon_cell_lonlat)

    # create geopandas dataframe for all these cells
    df_cell_lonlat = gpd.GeoDataFrame(pd.DataFrame({'x':[ix for iy in range(grid.dims['y'])],'y':[iy for iy in range(grid.dims['y'])]}),
                                      crs='epsg:4326',
                                      geometry=list_polygon_cell_lonlat)

    # convert it to the ESPF:3035 x/y projection (the one of the shapefile)
    df_cell = df_cell_lonlat.to_crs(crs_3035)

    # compute area in km2
    output = df_cell.area.values/1e6 #(m2 -> km2)

    return output



def SplitArray(x,n_per_group=None,n_groups=None):
    """ Split an array in several groups. 
    Use "n_per_group" or "n_groups" but not both.

    Parameters 
    ----------       
    x : numpy array
        Array to be splitted

    n_per_group : int
        Number of elements per group

    n_groups : int
        Number of groups

    Outputs
    -------

    output : list of list
        Groups

    Examples
    --------
    >>> from pyeslib import pyeslib
    >>> print(pyeslib.SplitArray(np.arange(10),n_per_group=3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> print(pyeslib.SplitArray(np.arange(10),n_per_group=8)) 
    [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]
    >>> print(pyeslib.SplitArray(np.arange(10),n_groups=4)) 
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    # load libraries
    import numpy as np

    # get number of elements per group
    if not n_groups is None:
        n_per_group = np.ceil(len(x)/n_groups)

    # loop
    istat, grp, grps = 0, [], []
    while istat <= len(x)-1:
        grp += [istat]
        if len(grp)==n_per_group:
            grps += [grp]
            grp = []
        istat += 1
    if len(grp)!=0: grps += [grp]
    return(grps)


def GetQuantileMappingParameters(x, training_obs, training_mod, nbins=100, npointsmin=10):
    import numpy as np
    qm_x, xbins, cdfobs, cdfmod = None, None, None, None

    w = np.where((np.isfinite(training_obs)==True) & (np.isfinite(training_mod)==True))[0]
    if len(w) > npointsmin:

        # Sort the arrays
        training_obs = np.sort(training_obs[w])
        training_mod = np.sort(training_mod[w])

        # Compute bins
        factor = 1
        maxxx = np.ceil(max(np.max(training_obs)*factor,np.max(training_mod)*factor))
        minnn = np.floor(min(np.min(training_obs)*factor,np.min(training_mod)*factor))
        xbins = np.arange(nbins)/(nbins-1)*(maxxx-minnn)+minnn  
        
        # Create PDF
        pdfobs, bins = np.histogram(training_obs, bins=xbins)
        pdfmod, bins = np.histogram(training_mod, bins=xbins)

        # Create CDF with zero in first entry.
        cdfobs = np.insert(np.cumsum(pdfobs), 0, 0.0)
        cdfmod = np.insert(np.cumsum(pdfmod), 0, 0.0)

        if not x is None:
            if GetType(x) in ['int','float']:
                if x >= maxxx:
                    # Set the max of observed concentrations
                    qm_x = np.nanmax(training_obs)
                else:
                    # Compute exact CDF values using linear interpolation
                    cdfx = np.interp(x, xbins, cdfmod, left=0.0, right=np.nan)
                    # Now use interpol again to invert the obsCDF, hence reversed x,y
                    qm_x = np.interp(cdfx, cdfobs, xbins, left=0.0, right=np.nan)
            else:
                print('...')
                qm_x = np.full([len(x)],np.nan)
                for ix in range(len(x)):
                    if x[ix] >= maxxx:
                        # Set the max of observed concentrations
                        qm_x = np.nanmax(training_obs)
                    else:
                        # Compute exact CDF values using linear interpolation
                        cdfx = np.interp(x[ix], xbins, cdfmod, left=0.0, right=np.nan)
                        # Now use interpol again to invert the obsCDF, hence reversed x,y
                        print([ix,len(x),np.interp(cdfx, cdfobs, xbins, left=0.0, right=np.nan)  ])
                        qm_x[ix] = np.interp(cdfx, cdfobs, xbins, left=0.0, right=np.nan)

        else:
            qm_x = None
    else:
        qm_x = np.full([len(x)], np.nan) if not x is None else None

    return((qm_x, xbins, cdfobs, cdfmod))                



def ChangeTimeScale(xtime, x, change, 
                    availability_criteria = {'criteria_d'       : 75,
                                             'criteria_h24'     : 75,
                                             'criteria_d8max'   : 75,
                                             'criteria_d1max'   : 75,
                                             'criteria_m'       : 75}):
    """ Change of the time scale.

    ChangeTimeScale computes the intput time series at a given time scale. From an 
    hourly time series, it can compute a daily 24-hour mean time series ("d"), a 
    daily 1-hour maximum time series ("d1max"), a daily 8-hour maximum time series 
    ("d8max"), a monthly mean time series ("m"). From a daily mean time series, it
    can compute a monthly mean time series ("m").
    It handles the application of a data availability criteria, expressed in %.

    Parameters 
    ----------       
    xtime : date array
        Date array at hourly or daily scale

    x : numpy.ndarray
        Arrays of values associated to xtime.

    change : str
        Change of time scale requested:
           "h_d" : from hourly to daily 24-h mea
           "h_d1max" : from hourly to daily 1-h maximum
           "h_d8max" : from hourly to daily 8-h maximum
           "h_m" : from hourly to monthly mean 
           "h_d_m" : from hourly to daily 24-h mean, and the from daily 24-h mean to monthly mean 
           "h_d1max_m" : from hourly to daily 1-h maximum, and then to monthly mean 
           "h_d8max_m" : from hourly to daily 8-h maximum, and then to monthly mean

    availability_criteria : dict
        Dictionnary specifying the criteria of data availability in % to be applied, 
        depending on the time scale requested (possible keys : "criteria_h24", "criteria_d", 
        "criteria_d1max", "criteria_d8max", "criteria_m").    

    Outputs
    -------

    output : numpy.ndarray
        Arrays of values at the new requested time scale

    Examples
    --------
    >>> from pyeslib import ChangeTimeScale
    >>> import datetime
    >>> import numpy as np
    >>> import pandas as pd
    >>> xtime = pd.date_range(datetime.datetime.strptime('2019020500', '%Y%m%d%H'),
    ...                       datetime.datetime.strptime('2019070123', '%Y%m%d%H'), freq='H')
    >>> x = np.arange(len(xtime))
    >>> print(ChangeTimeScale(xtime, x, change='h_d1max'))
        [  23.   47.   71.   95.  119.  143.  167.  191.  215.  239.  263.  287.
          311.  335.  359.  383.  407.  431.  455.  479.  503.  527.  551.  575.
          599.  623.  647.  671.  695.  719.  743.  767.  791.  815.  839.  863.
          887.  911.  935.  959.  983. 1007. 1031. 1055. 1079. 1103. 1127. 1151.
         1175. 1199. 1223. 1247. 1271. 1295. 1319. 1343. 1367. 1391. 1415. 1439.
         1463. 1487. 1511. 1535. 1559. 1583. 1607. 1631. 1655. 1679. 1703. 1727.
         1751. 1775. 1799. 1823. 1847. 1871. 1895. 1919. 1943. 1967. 1991. 2015.
         2039. 2063. 2087. 2111. 2135. 2159. 2183. 2207. 2231. 2255. 2279. 2303.
         2327. 2351. 2375. 2399. 2423. 2447. 2471. 2495. 2519. 2543. 2567. 2591.
         2615. 2639. 2663. 2687. 2711. 2735. 2759. 2783. 2807. 2831. 2855. 2879.
         2903. 2927. 2951. 2975. 2999. 3023. 3047. 3071. 3095. 3119. 3143. 3167.
         3191. 3215. 3239. 3263. 3287. 3311. 3335. 3359. 3383. 3407. 3431. 3455.
         3479. 3503. 3527.]
    """
    # Load libraries
    import numpy as np
    import pandas as pd
    import warnings
    
    # Check type of inputs)
    #assert GetType(xtime) in ['pandas.core.indexes.datetimes.DatetimeIndex',
    #                          'pd.core.indexes.datetimes.DatetimeIndex'],'xtime should be a pandas datetime index'
    assert GetType(x) in ['numpy.ndarray','np.ndarray'],'x should be a numpy array'
    assert GetType(change)=='str','change should be a string'
    assert GetType(availability_criteria)=='dict','availability_criteria should be a dictionnary'

    if change=='h_h24':
        # Compute diurnal profile
        output = np.full([24], np.nan)
        try:
            xtime_h = xtime.strftime('%H') #(faster but not always possible) 
        except:
            xtime_h = np.array([pd.to_datetime(i).strftime('%H') for i in xtime])
        for ih in np.arange(24):
            w = np.where((xtime_h=='{:02d}'.format(ih)) & (np.isfinite(x)))[0]
            if float(len(w)) >= availability_criteria['criteria_h24']/100:
                output[ih] = np.nanmean(x[w])

    elif change in ['h_d','h_d1max','h_d8max']:
        
        # Compute number of days
        ndays = int(len(x)/24)
        
        criteria = availability_criteria[{'h_d':'criteria_d','h_d1max':'criteria_d1max','h_d8max':'criteria_d8max'}[change]]

        if change in ['h_d','h_d1max']:
            # Get data masked according to data availability criteria            
            mask = np.array([len(np.where(np.isfinite(x[iday*24:iday*24+24]))[0]) >= 24*criteria/100 for iday in range(ndays)])+0.
            w = np.where(mask==0)
            if len(w[0])!=0: mask[w] = np.nan
            x_masked = np.multiply(x.reshape(ndays,24).transpose(),mask)
            
            # Change time scale
            if change=='h_d':
                # Compute daily 24-hours mean time series
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    output = np.nanmean(x_masked,axis=0)
            elif change=='h_d1max':
                # Compute daily 1-hour maximum time series  
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    output = np.nanmax(x_masked,axis=0)
                    
        elif change=='h_d8max':
            window = 8
            # Reshape input data
            x_reshaped = x.reshape(ndays,24).transpose()
            x_dayminusone = np.concatenate((np.full([24],np.nan),x[:(ndays-1)*24])) 
            x_dayminusone_reshaped = x_dayminusone.reshape(ndays,24).transpose()
            x_reshaped_extended = np.concatenate((x_dayminusone_reshaped,x_reshaped),axis=0)
            
            # Loop on 8-hour windows
            x_masked_window = np.full([24,ndays],np.nan)
            for ih in np.arange(24):
                # Consider a specific 8-hour window 
                x_window = x_reshaped_extended[24+ih-window+1:24+ih+1,:]
                # Compute the 8-hour average
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    x_masked_window[ih,:] = np.nanmean(x_window,axis=0)
                
                # Identify days when there was enough hourly data over that specific 8-hour window (assign NaN elsewhere)
                mask_window = np.array([len(np.where(np.isfinite(x_window[:,iday]))[0])/window*100 >= criteria for iday in range(ndays)])+0.  
                w = np.where(mask_window==0)
                if len(w[0])!=0: mask_window[w] = np.nan
                # Update the 8-hour average
                x_masked_window[ih,:] = np.multiply(x_masked_window[ih,:],mask_window)
                
            # Identify days where there was enough 8-hourly average (assign NaN elsewhere)
            mask = np.array([len(np.where(np.isfinite(x_masked_window[:,iday]))[0])/24*100 >= criteria for iday in range(ndays)])+0.
            w = np.where(mask==0)
            if len(w[0])!=0: mask[w] = np.nan
            
            # Update 
            x_masked_window = np.multiply(x_masked_window,mask)

            # Compute the daily maximum 8-hour average
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                output = np.nanmax(x_masked_window,axis=0)
                
            '''
            # OLD VERSION:
            x_masked_window = np.full([24-window+1,ndays],np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for ih in np.arange(0,24-window+1,1):
                    x_masked_window[ih,:] = np.nanmean(x_masked[ih:ih+window,:],axis=0)
                output = np.nanmax(x_masked_window,axis=0)
            '''
            
    elif change=='h_m':
        # Compute monthly average time series from hourly time series
        try:
            xtime_yyyymm = xtime.strftime('%Y-%m') #(faster but not always possible)  
        except:
            xtime_yyyymm = np.array([i.strftime('%Y-%m') for i in xtime])
        xtime_yyyymm_unique = np.unique(xtime_yyyymm)
        output = np.full([len(xtime_yyyymm_unique)], np.nan)
        for i in np.arange(len(xtime_yyyymm_unique)):
            w = np.where((xtime_yyyymm==xtime_yyyymm_unique[i]) & (np.isfinite(x)))[0]
            if float(len(w)) >= 24*availability_criteria['criteria_m']/100:
                output[i] = np.nanmean(x[w])

    elif change=='d_m':
        # Compute monthly average time series from daily 24-hour mean time series
        try:
            xtime_yyyymm = xtime.strftime('%Y-%m') #(faster but not always possible) 
        except:
            xtime_yyyymm = np.array([i.strftime('%Y-%m') for i in xtime])
        xtime_yyyymm_unique = np.unique(xtime_yyyymm)
        output = np.full([len(xtime_yyyymm_unique)], np.nan)
        for i in np.arange(len(xtime_yyyymm_unique)):
            w = np.where((xtime_yyyymm==xtime_yyyymm_unique[i]) & (np.isfinite(x)))[0]
            if float(len(w)) >= availability_criteria['criteria_m']/100:
                output[i] = np.nanmean(x[w])

    elif change in ['h_d_m','h_d1max_m','h_d8max_m']:
        # Compute monthly average time series from daily 24-hour mean time series
        import pandas as pd

        # (first, change from hourly to d/d1max/d8max)
        if change=='h_d_m':
            y = ChangeTimeScale(xtime, x, change='h_d', availability_criteria=availability_criteria)
        elif change=='h_d1max_m':
            y = ChangeTimeScale(xtime, x, change='h_d1max', availability_criteria=availability_criteria)
        elif change=='h_d8max_m':
            y = ChangeTimeScale(xtime, x, change='h_d8max', availability_criteria=availability_criteria)

        # (compute daily time array)
        ytime = np.array([pd.to_datetime(i) for i in np.unique(xtime.strftime('%Y-%m-%d'))])

        # (second, change from daily to monthly)
        output = ChangeTimeScale(ytime, y, change='d_m', availability_criteria=availability_criteria)

    return(output)

def ComputeFairemodeMQI(o,m,pollutant='O3'):
    import numpy as np
    
    beta = 2
    if pollutant=='O3':
        u_95r_RV = 0.18
        RV = 120 #(ug/m3)                                                                                                            
        alpha = 0.79
        N_p = 11
        N_np = 3

    wok = np.where(np.isfinite(o+m)==True)[0]
    if len(wok)!=0:
        # Remove missing data                                                                                                        
        o,m = o[wok],m[wok]

        #mqi = np.full([len(o)],np.nan)                                                                                              
        #u_95 = np.full([len(o)],np.nan)                                                                                             
        #for i in wok:                                                                                                               
        #    u_95[i] = u_95r_RV*np.sqrt((1-alpha**2)*o[i]**2 + (alpha*RV)**2)                                                        
        #    mqi[i] = np.abs(m[i]-o[i])/(beta*u_95[i])                                                                               
        #rms_u_recalculated = np.sqrt(np.nansum(u_95**2)/len(wok))                                                                   

        rms_u = u_95r_RV*np.sqrt((1-alpha**2)*(np.nanmean(o)**2-np.nanstd(o)**2) + (alpha*RV)**2)
        rmse = np.sqrt(np.mean((m-o)**2)) 
        mqi = rmse/(beta*rms_u)
        #print('RMSu={:8.2f}  RMSE={:8.2f} => MQI={:8.2f}'.format(rms_u,rmse,mqi))
        return(mqi)
    else:
        return(np.nan)



    
def ComputeStatistics(obs, mod, metrics, fmt='8.2f', print_result=False):
    """ Computation of statistical metrics.

    ComputeStatistics performs a statistical evaluation between two arrays, considering a set 
    of continuous and/or categorical statistical metrics. Potential data gaps in both arrays
    are harmonized before computing the statistical metrics, which ensures a fair comparison 
    whatever for all metrics.

    Parameters
    ----------

    obs : numpy.ndarray
        Primary numpy array of data, to be used as the reference.

    mod : numpy.ndarray  
        Secondary numpy array, to be evaluated against the primary one.

    metrics : list of strings
        List of statistical metrics to be computed.

    fmt : string, default="8.2f"
        Format for the output results. If float requested, integer metrics
        will be forced to have zero digits. To be properly interpreted, both a
        dot and a letter are required in this parameter. 

    print_result : boolean, default=False
        Flag to print the formatted statistical results

    Outputs
    ----------

    output : dict
        Dictionnary containing :
             - result     : numpy.ndarray with numerical results
             - result_str : str with formatted results
             - header_str : str with formatted header containing the metrics
             - result_str_latex : str with formatted results, in Latex-table-format
             - header_str_latex : str with formatted header containing the metrics, in Latex-table-format

    Examples
    --------
    >>> from pyeslib import pyeslib
    >>> import numpy as np
    >>> obs = np.arange(100)
    >>> mod = np.arange(100)**2
    >>> metrics = ['MB','nMB','RMSE','nRMSE','PCC','N']
    >>> res = pyeslib.ComputeStatistics(obs, mod, metrics) 
    >>> print(res['result'])
    [3.23400000e+03 6.53333333e+03 4.36079887e+03 8.80969469e+03
    9.67644393e-01 1.00000000e+02]
    >>> print('{}\n{}'.format(res['header_str'],res['result_str']))
          MB     nMB    RMSE   nRMSE     PCC       N
     3234.00 6533.33 4360.80 8809.69    0.97     100
    >>> print('{}\n{}'.format(res['header_str_latex'],res['result_str_latex']))
          MB &      nMB &     RMSE &    nRMSE &      PCC &        N \\ 
     3234.00 &  6533.33 &  4360.80 &  8809.69 &     0.97 &      100 \\ 
    """
    
    # Load libraries
    import numpy as np
    import scipy.stats

    # Check type of input arguments
    assert GetType(obs)=='numpy.ndarray','obs should be a numpy array'
    assert GetType(mod)=='numpy.ndarray','mod should be a numpy array'
    assert GetType(metrics) in ['list','numpy.ndarray','np.ndarray'],'metrics should be a list or numpy array'
    assert GetType(metrics[0])=='str','metrics components should be strings'
    assert GetType(fmt)=='str','fmt should be a string'
    assert GetType(print_result) in ['NoneType','bool'],'print_result should be a boolean or None'
    assert fmt[-1].isalpha()==True and '.' in fmt,'fmt not properly defined'    
        
    # Define result array
    nmetrics = len(metrics)
    result = np.full([nmetrics],np.nan)
        
    # Get comparable finite subsets
    w = np.where((np.isfinite(obs)==True) & (np.isfinite(mod)==True))[0]

    # Get number of points available
    if 'N' in metrics:
        result[np.where(np.array(metrics)=='N')[0]] = len(w)

    # If at least one point, compute statistics
    if len(w)!=0:
        o = obs[w]
        m = mod[w]

        # Precompute linear regression if some related metrics are requested
        if 'slope' in metrics or 'inter' in metrics or 'p_val' in metrics:
            slope, inter, r_val, p_val, std_err = scipy.stats.linregress(o,m)
           
        # Loop on the desired metrics
        threshold_previous = None
        for imet,xmet in enumerate(metrics):

            met = xmet.split('|')[0]
            if len(xmet.split('|'))==2:
                threshold = float(xmet.split('|')[1])
                # Precompute contingency table and some basic detection metrics (to be used in some other formulas)   
                if threshold != threshold_previous:
                    o_boolean = (o >= threshold)
                    m_boolean = (m >= threshold)
                    a = len(np.where((o_boolean==True ) & (m_boolean==True ))[0])
                    b = len(np.where((o_boolean==False) & (m_boolean==True ))[0])
                    c = len(np.where((o_boolean==True ) & (m_boolean==False))[0])
                    d = len(np.where((o_boolean==False) & (m_boolean==False))[0])
                    s = (a+c)/(a+b+c+d)
                    n = a+b+c+d
                    h = a/(a+c) if a+c!=0 else np.nan
                    f = b/(b+d) if b+d!=0 else np.nan
                    h_error = h*(1-h)/(a+c) if a+c!=0 else np.nan
                    f_error = f*(1-f)/(b+d) if b+d!=0 else np.nan
                    ar = (a+b)*(a+c)/n
                    dr = (b+d)*(c+d)/n
            
            # Continuous metrics
            if met=='MB': #(mean bias)
                result[imet] = np.mean(m-o)
            elif met=='nMB' and np.mean(o)!=0: #(normalized mean bias)
                result[imet] = np.mean(m-o)/np.mean(o)*100
            elif met=='MNB': #(mean normalized bias)
                result[imet] = np.mean((m-o)/o*100)
            elif met=='MSE': #(mean square error)
                result[imet] = np.mean((m-o)**2)
            elif met=='nMSE' and np.mean(o)!=0: #(normalized mean square error)
                result[imet] = np.mean((m-o)**2)/np.mean(o)*100
            elif met=='RMSE': #(root mean square error)
                result[imet] = np.sqrt(np.mean((m-o)**2))
            elif met=='nRMSE' and np.mean(o)!=0: #(normalized root mean square error)
                result[imet] = np.sqrt(np.mean((m-o)**2))/np.mean(o)*100
            elif met=='PCC' and len(o)>1 : #(Pearson correlation coefficient)
                result[imet] = scipy.stats.pearsonr(o,m)[0]
            elif met=='PCC2' and len(o)>1 : #(squared Pearson correlation coefficient)
                result[imet] = scipy.stats.pearsonr(o,m)[0]**2
            elif met=='R-squared' and len(o)>1 : #(coefficient of determination)
                result[imet] = 1.0 - np.sum((o-m)**2)/np.sum((o-np.mean(o))**2)
            elif met=='nMSDB' and len(o)>1 and np.nanstd(o)!=0: #(normalized mean standard deviation bias)
                result[imet] = (np.nanstd(m)-np.nanstd(o))/np.nanstd(o)*100
            elif met=='RMSEu': #(unsystematic root mean square error)
                result[imet] = np.sqrt( np.mean( (m-np.mean(m)-o+np.mean(o))**2))
            elif met=='nRMSEu' and np.mean(o)!=0: #(normaized unsystematic root mean square error)    
                result[imet] = np.sqrt( np.mean( (m-np.mean(m)-o+np.mean(o))**2))/np.mean(o)*100
            elif met=='MAE': #(mean absolute error)
                result[imet] = np.mean(np.abs(m-o)) 
            elif met=='nMAE' and np.mean(o)!=0: #(normalized mean absolute error) 
                result[imet] = np.mean(np.abs(m-o))/np.mean(o)*100
            elif met=='MNAE': #(mean normalized absolute error) 
                result[imet] = np.mean(np.abs(m-o) / o)
            elif met=='MFB': #(mean fractional bias) 
                result[imet] = np.mean((m-o) / ((m+o) / 2.))
            elif met=='MAFB': #(mean absolute fractional bias) 
                result[imet] = np.mean(np.abs((m-o) / ((m+o) / 2.)))
            elif met=='FAC2': #(fraction of model values within a factor of 2 of observed values)
                frac = m / o
                result[imet] =  (100.0 / len(frac)) * len(frac[(frac >= 0.5) & (frac <= 2.0)])
            elif met=='UPA': #(unpaired peak accuracy)
                #(from https://gitlab.com/polyphemus/atmopy/-/blob/master/stat/measure.py)
                result[imet] = (np.max(m) - np.max(o)) / np.max(o)
            elif met.startswith('E') and met[1:].isnumeric()==True: #(coefficient of efficiency Ej with j=1 or 2
                # (from Legates DR, McCabe GJ. (2012). A refined index of model performance: a rejoinder. International Journal of Climatology)
                result[imet] = 1.0 - np.sum(np.abs(m-o)**j)/np.sum(np.abs(o-np.mean(o))**j)
            elif met=='IOA': #(index of agreement)
                #(from Willmott, C.J., Robeson, S.M., Matsuura, K., 2011. A refined index of model performance. International Journal of Climatology)
                result[imet] = 1.0 - (np.sum((o - m) ** 2)) / (np.sum((np.abs(m-np.mean(o)) + np.abs(o-np.mean(o))) ** 2))
            elif met=='%N': #(percentage of data available)
                result[imet] = len(o)/len(obs)*100
            elif met=='Mo': #(mean observation)
                result[imet] = np.mean(o)
            elif met=='Mm': #(mean model)
                result[imet] = np.mean(m)
            elif met=='So': #(standard deviation observation)
                result[imet] = np.std(o)
            elif met=='Sm': #(standard deviation model)
                result[imet] = np.std(m)
            elif met=='slope': #(linear regression slope)
                result[imet] = slope
            elif met=='inter': #(linear regression intercept)
                result[imet] = inter
            elif met=='p_val': #(linear regresion p-value)
                result[imet] = p_val
            elif met.startswith('P|'): #(percentile of the difference)
                percentile = float(met[2:])
                result[imet] = np.percentile(m-o,percentile)
            elif met=='MQI':   #(FAIRMODE modeling quality indicator)
                result[imet] = ComputeFairemodeMQI(o,m)


                
            # Categorical metrics
            if met=='a': #(number of true positives)
                result[imet] = a
            elif met=='b': #(number of false positives)
                result[imet] = b
            elif met=='c': #(number of false negatives)
                result[imet] = c
            elif met=='d': #(number of true negatives)
                result[imet] = d
            elif met=='a+c': #(number of exceedances observed)
                result[imet] = a+c
            elif met=='a+b': #(number of exceedances modeled)
                result[imet] = a+b
            elif met=='s': #(base rate)
                result[imet] = s
            elif met=='n': #(total number of exceedances)
                result[imet] = n
            elif met=='H': #(hit rate)
                result[imet] = h
            elif met=='F': #(false alarm rate)
                result[imet] = f
            elif met=='SR': #(success ratio)
                result[imet] = a/(a+b) if a+b!=0 else np.nan
            elif met=='SRN': #(success ratio negative)
                result[imet] = d/(c+d) if c+d!=0 else np.nan
            elif met=='CSI': #(critical success index)
                result[imet] = a/(a+b+c) if a+b+c!=0 else np.nan
            elif met=='PSS': #(Peirce skill score)
                result[imet] = a/(a+c)-b/(b+d) if a+c!=0 and b+d!=0 else np.nan
            elif met=='GSS': #(Gilbert skill score)
                result[imet] = (a-ar) / (a+b+c-ar) if a+b+c-ar!=0 else np.nan
            elif met=='HSS': #(Heidke skill score)
                result[imet] = (a+d-ar-dr) / (n-ar-dr) if n-ar-dr!=0 else np.nan
            elif met=='FAR': #(false alarm ratio)
                result[imet] = b/(a+b) if a+b!=0 else np.nan
            elif met=='PC': #(percentage of correct)
                result[imet] = (a+d)/n
            elif met=='FB': #(frequency bias)
                result[imet] = (a+b)/(a+c) if a+c!=0 else np.nan
            elif met=='ORSS': #(odds ratio skill score)
                result[imet] = (a*d-b*c)/(a*d+b*c) if a*d+b*c!=0 else np.nan                
            elif met=='AUC': #( area under the ROC curve)
                import sklearn.metrics
                result[imet] = sklearn.metrics.roc_auc_score(o_boolean,m) if len(np.unique(o_boolean))==2 else np.nan

    # Retrieve format information
    fmt_type = fmt[-1]
    fmt_width = fmt.split('.')[0]
    fmt_digit = fmt[:-1].split('.')[1]

    # Get formatted string results
    result_string,header_string = '',''
    result_string_latex,header_string_latex = '',''
    for imet,met in enumerate(metrics):
        
        # Define the Latex column separator
        sep = '\\\\' if met==metrics[-1] else '&'
        
        # Adjust the number of digits if integer metric and float requested
        digit = fmt_digit
        if met in ['a','b','c','d','a+b','a+c','n','N','%N','nMB','nRMSE','nMSDB'] and fmt_type=='f': digit = '0'
        if met in ['MB','RMSE','Mo','Mm'] and fmt_type=='f': digit = '1'
        if met in ['PCC','slope'] and fmt_type=='f': digit = '2'
        
        # Get the formatted results and headers
        if np.isfinite(result[imet]):
            result_string += ('{:'+fmt_width+'.'+digit+fmt_type+'}').format(result[imet])
            result_string_latex += ('{:'+fmt_width+'.'+digit+fmt_type+'} '+sep+' ').format(result[imet])
        else:
            result_string += ('{:>'+fmt_width+'s}').format('nan')
            result_string_latex += ('{:>'+fmt_width+'s} '+sep+' ').format('nan')
        header_string += ('{:>'+fmt_width+'s}').format(met)
        header_string_latex += ('{:>'+fmt_width+'s} '+sep+' ').format(met)

    # Print results
    if print_result==True:
        print('{}\n{}'.format(header_string,result_string))

    output = {'result':result,
              'result_str':result_string,
              'header_str':header_string,
              'result_str_latex':result_string_latex,
              'header_str_latex':header_string_latex}
    
    return(output)


def ConvertRegularLonLatIntoGrid(xlon,xlat,digits=5):
    """ Convert the longitude and latitude of the centers of grid cells
    into a curvilinear-like grid xarray dataset (useful for instance for
    regridding data to another curvilinear or non-curvilinear grid with xESMF).

    Parameters  
    ----------  
    xlon : numpy array
        Longitudes of the center of the grid cells

    xlat : numpy array
        Latitudes of the center of the grid cells

    digits : integer
        Number of digits to consider for rounding numbers

    Output  
    ------
    output : xarray dataset with
       - lon : longitude of the center of the cell
       - lat : latitude of the center of the cell
       - lon_b : longitude of the bounds (corners) of the cell
       - lat_b : latitude of the bounds (corners) of the cell    
       - attributes :
            - reslon : resolution on the longitude
            - reslat : resolution on the latitude
            - extent : geographical extent, expressed as [lon0,lon1,lat0,lat1]
       
    Example  
    -------  
    """
    # Load libraries
    import numpy as np
    import xarray as xr
    from datetime import datetime
    
    # Check type of input arguments
    assert GetType(xlon) in ['numpy.ndarray','np.ndarray'],'xlon should be a numpy array'
    assert GetType(xlat) in ['numpy.ndarray','np.ndarray'],'xlat should be a numpy array'

    # Get number of grid cells
    nx,ny = len(xlon),len(xlat)

    # Compute longitude and latitude resolution (assumed to be constant and uniform)
    reslon = np.round(np.nanmean(xlon[1:]-xlon[:-1]),digits)
    reslat = np.round(np.nanmean(xlat[1:]-xlat[:-1]),digits)

    # Compute lon/lat of grid bounds
    xlon_b = np.full([nx+1],np.nan)
    for ix in range(nx+1):
        xlon_b[ix] = np.round(xlon[ix] - reslon/2,digits) if ix<nx else np.round(xlon[-1] + reslon/2,digits)
    xlat_b = np.full([ny+1],np.nan)
    for iy in range(ny+1):
        xlat_b[iy] = np.round(xlat[iy] - reslat/2,digits) if iy<ny else np.round(xlat[-1] + reslat/2,digits)
            
    # Create longitude and latitude mesh grids
    lon, lat = np.meshgrid(xlon, xlat)
    lon_b, lat_b = np.meshgrid(xlon_b, xlat_b)

    # Compute geographical extent of the grid
    extent = [np.nanmin(lon_b),np.nanmax(lon_b),np.nanmin(lat_b),np.nanmax(lat_b)]
    
    # Create a grid xarray dataset
    output = xr.Dataset({'lon': (['y','x'], lon),
                         'lat': (['y','x'], lat),
                         'lon_b': (['y_b','x_b'], lon_b),
                         'lat_b': (['y_b','x_b'], lat_b)})

    # Add basic attributes
    output.attrs.update({'grid_type'             : 'longitude-latitude regular grid',
                         'extent'                : extent,
                         'resolution_degree_lon' : reslon,
                         'resolution_degree_lat' : reslat,
                         'creation_date':str(datetime.today())})

    return(output)


def ConvertUnits(x,change,species):
    """ Convert units

    Parameters  
    ----------  
    x : float or numpy array
        Concentration at the original unit

    change : string
        String indicating the desired unit changed written such as "<unit_origin>|<unit_final>"

    species : string
        Chemical species

    Output  
    ------
    
    x_converted : float
       Concentrations "x" converted to the desired unit

    unit_factor : float
       Convertion factor used
    
    Example  
    -------  
    """

    # Check type of input arguments
    assert GetType(x) in ['float',
                          'numpy.ndarray','np.ndarray',
                          'xarray.core.dataarray.DataArray','xr.core.dataarray.DataArray'],\
                          'x should be a float or a numpy array'
    assert GetType(change)=='str','change should be a string'
    assert GetType(species)=='str','species should be a string'
    
    # Define units naming dictionnary
    units_dict = {'ppbV'  :['1e-9','ppbv','ppbV','ppb','nmole/mole','nmole mole-1'],
                  'ppmV'  :['ppmv','ppmV','ppm'],
                  'ug m-3':['ug m-3','ug m**-3','ugm-3','ug/m3','microg/m3','microg m-3','microgm-3','µg/m3','µgm-3','µg m-3'],
                  'mg m-3':['mg m-3','mg m**-3','mgm-3','mg/m3'],
                  'g m-3' :['g m-3','g m**-3','gm-3','g/m3'],
                  'kg m-3':['kg m-3','kg m**-3','kgm-3','kg/m3','kilog/m3','kilog m-3','kilogm-3'],
                  'kg kg-1':['kg kg-1','kgkg-1','kg/kg','kilog/kilog','kg kg**-1']}
    
    # Read origin unit and final unit requested
    unit_origin_specified = change.split('|')[0]
    unit_final_specified = change.split('|')[1]

    # Handle differences of writing of the units
    for key,value in units_dict.items():
        if unit_origin_specified in value:
            unit_origin = key
        if unit_final_specified in value:
            unit_final = key

    # Print message
    print('converting from {} (reformatted to {}) to {} (reformatted to {})'.\
          format(unit_origin_specified,unit_origin,unit_final_specified,unit_final))
            
    # Define molar masses in g/mol (source : https://pubchem.ncbi.nlm.nih.gov/)
    molar_mass_dict = {'sconco3' :47.9982,
                       'sconcno2':46.0055,
                       'sconcso2':64.0638,
                       'sconcco' :28.0101,
                       'sconcno' :30.0061}
    
    # Compute temporary factor
    # EEA measurements these are specifically defined as for 293 K temperature, and 1013 hPa for pressure here for all gas-phase compounds:
    # Source : https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32008L0050&from=en
    #          https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32004L0107&from=EN 
    if species in list(molar_mass_dict.keys()):
        #gas_constant = 8.31446261815324
        #pressure_reference_in_pascal = 101325 
        temperature_reference_in_degree_celsius = 20
        factor = 12.187*molar_mass_dict[species] / (273.15+temperature_reference_in_degree_celsius)
        molar_mass_dry_air = 28.9644
        
    # Convert units
    if unit_origin==unit_final:
        unit_factor = 1.
        x_converted = x
    else:
        if unit_origin=='ppbV' and unit_final=='ug m-3'  : unit_factor = factor
        if unit_origin=='ppbV' and unit_final=='ppmV'    : unit_factor = 1/1000
        if unit_origin=='ppmV' and unit_final=='ppbV'    : unit_factor = 1000
        if unit_origin=='ppmV' and unit_final=='ug m-3'  : unit_factor = factor*1000
        if unit_origin=='ug m-3' and unit_final=='ppbV'  : unit_factor = 1/factor
        if unit_origin=='ug m-3' and unit_final=='ppmV'  : unit_factor = 1/(factor*1000)
        if unit_origin=='mg m-3' and unit_final=='ug m-3': unit_factor = 1e3
        if unit_origin=='g m-3'  and unit_final=='ug m-3': unit_factor = 1e6
        if unit_origin=='kg m-3' and unit_final=='ug m-3': unit_factor = 1e9
        if unit_origin=='kg m-3' and unit_final=='ppbV'  : unit_factor = 1e9/factor
        if unit_origin=='kg m-3' and unit_final=='ppmV'  : unit_factor = 1e9/(factor*1000)
        if unit_origin=='kg kg-1' and unit_final=='ppbV' : unit_factor = molar_mass_dry_air/molar_mass_dict[species]*1e9
        if unit_origin=='kg kg-1' and unit_final=='ppmV' : unit_factor = molar_mass_dry_air/molar_mass_dict[species]*1e6



        
        try:
            x_converted = x*unit_factor
        except:
            print(['ISSUE',x,change,species,unit_origin,unit_final])
            x_converted = x*unit_factor 
                        
    return((x_converted,unit_factor))

    
def DefineLonLatRegularGrid(extent, reslon, reslat):
    """ Create an xarray dataset with regular longitude-latitude
    grid over the prescribed geographical extent

    Parameters  
    ----------  
    extent : list of integers/floats
        Geographical extent, expressed as [lon0,lon1,lat0,lat1]

    reslon : float
        Resolution on the longitude

    reslat : float
        Resolution on the latitude

    Output  
    ------
    
    output : xarray dataset with
       - lon : longitude of the center of the cell
       - lat : latitude of the center of the cell
       - lon_b : longitude of the bounds (corners) of the cell
       - lat_b : latitude of the bounds (corners) of the cell    
    
    Example  
    -------  
    >>> from pyeslib import pyeslib                       
    >>> extent,reslon,reslat = [-10., 5., 35., 45.], 1, 1
    >>> grid = pyeslib.DefineLonLatRegularGrid(extent, reslon, reslat) 
    >>> print(grid)
    <xarray.Dataset>
    Dimensions:  (y: 9, x: 14, y_b: 10, x_b: 15)
    Dimensions without coordinates: y, x, y_b, x_b
    Data variables:
        lon      (y, x) float64 -9.5 -8.5 -7.5 -6.5 -5.5 ... -0.5 0.5 1.5 2.5 3.5
        lat      (y, x) float64 35.5 35.5 35.5 35.5 35.5 ... 43.5 43.5 43.5 43.5
        lon_b    (y_b, x_b) float64 -10.0 -9.0 -8.0 -7.0 -6.0 ... 1.0 2.0 3.0 4.0
        lat_b    (y_b, x_b) float64 35.0 35.0 35.0 35.0 35.0 ... 44.0 44.0 44.0 44.0
    Attributes:
        grid_type:              longitude-latitude regular grid
        extent:                 [-10.0, 5.0, 35.0, 45.0]
        resolution_degree_lon:  1
        resolution_degree_lat:  1
        creation_date:          2022-01-04 16:21:57.682555
    """
    # Load libraries
    import numpy as np
    import xarray as xr
    from datetime import datetime, timedelta
    import copy

    # Check type of inputs
    assert GetType(extent)=='list','extent should be a list'
    assert GetType(extent[0]) in ['int','float','numpy.float64','np.float64','numpy.int64','np.int64'],'extent components should be int or float'
    assert len(extent)==4,'extent should have 4 components'
    assert GetType(reslon) in ['int','float'], 'reslon should an int or float'
    assert GetType(reslat) in ['int','float'], 'reslat should an int or float'
    
    # Prepare longitudes
    round_digits = 5
    xlon_b = []
    x = np.round(copy.deepcopy(extent[0]), round_digits)
    while np.round(x, round_digits) <= extent[1]:
        xlon_b.append(np.round(x, round_digits))
        x += reslon
    xlon = xlon_b[:-1] + np.round(reslon/2., round_digits)

    # Prepare latitudes
    xlat_b = []
    x = np.round(copy.deepcopy(extent[2]), round_digits)
    while np.round(x, round_digits) <= extent[3]:
        xlat_b.append(np.round(x, round_digits))
        x += reslat
    xlat = xlat_b[:-1] + np.round(reslat/2., round_digits)

    # Create longitude and latitude mesh grids
    lon, lat = np.meshgrid(xlon, xlat)
    lon_b, lat_b = np.meshgrid(xlon_b, xlat_b)

    # Create a grid xarray dataset
    output = xr.Dataset({'lon': (['y','x'], lon),
                         'lat': (['y','x'], lat),
                         'lon_b': (['y_b','x_b'], lon_b),
                         'lat_b': (['y_b','x_b'], lat_b)})
    
    # Add basic attributes
    output.attrs.update({'grid_type'             : 'longitude-latitude regular grid',
                         'extent'                : extent,
                         'resolution_degree_lon' : reslon,
                         'resolution_degree_lat' : reslat,
                         'creation_date':str(datetime.today())})

    return(output)


def GetGHOSTData(htime=None,
                 stations_info=None,
                 pollutant=None,
                 path_ghost=None,
                 drop_qa=None,
                 n_jobs=1,
                 timescale=['hourly'],
                 verbose=False):
    """ Get GHOST data over a given time period

    GetGHOSTDataMonth extract the GHOST pollutant concentrations 
    at hourly scale, for a given set of stations (specified in 
    stations_info), applying the Quality-Assurance filtering 
    (removing the data having at least one of the flags specified
    in drop_qa).

    Parameters
    ----------

    yyyymm : str
        Year and month (in YYYYMM format)
    
    htime : date array
        Entire hourly date array of interest. Only useful if the user is not interested
        entire months.        
    
    stations_info : list or pandas dataframe
        Either a list of stations of interest, or a pandas dataframe including 
        the stations of interest (column "stations' mandatory) with additional
        information (e.g. country, longitude, latitude...). If specified, all 
        this extra information will be included in the output dataset.

    pollutant : str
        Pollutant requested. Should follow the GHOST nomenclatura.

    path_ghost : str
        Path of the GHOST root directory

    drop_qa : list of int/float, default=None 
        List of Quality-Assurance (QA) flags to filter. All measurements with one of 
        these flags will be removed

    n_jobs : int, default=1
        Number of jobs to run in parallel. If n_jobs > 1, then the function will
        run in parallel, each month being run in parallel on a specific job.

    verbose : bool, default=False
        To print intermediate information

    Output
    ------
    output : xarray dataset with
        - hconc (hourly concentrations at requested stations)
        As well as (if information sent in stations_info):
        - longitude  
        - latitude  
        - altitude   
        - station_classification 
        - area_classification
        - country

    Example
    -------
    >>> from pyeslib import pyeslib
    >>> 
    >>> extent,reslon,reslat = [-10., 5., 35., 45.], 1, 1
    >>> time1, time2 = '2018010100','2018033123'
    >>> network = 'EEA_AQ_eReporting'
    >>> ghost_version = '1.3.3'
    >>> path_ghost='/gpfs/projects/bsc32/AC_cache/obs/ghost/{}/{}'.format(network,ghost_version)
    >>> drop_qa = [0, 1, 2, 3, 6, 8, 10, 12, 13, 14, 17, 18, 22, 25, 40, 41, 42]
    >>> station_classification=['background', 'point_source-industrial']
    >>> area_classification=['rural']
    >>> n_jobs = 12
    >>> 
    >>> timeinfo = pyeslib.PrepareTimeArrays(time1, time2, time_scales=['h','d','m','y'])
    >>> stations_info = \
    ...     pyeslib.SelectGHOSTStations(timeinfo['htime'],
    ...                                 pollutant='sconco3',
    ...                                 path_ghost=path_ghost,
    ...                                 extent=extent,
    ...                                 drop_qa=drop_qa,
    ...                                 station_classification=station_classification,
    ...                                 area_classification=area_classification,
    ...                                 n_jobs=n_jobs)
    >>> ghost_ds = \
    ...     pyeslib.GetGHOSTData(timeinfo['htime'],
    ...                          stations_info,
    ...                          pollutant='sconco3',
    ...                          path_ghost=path_ghost,
    ...                          drop_qa=drop_qa,
    ...                          n_jobs=n_jobs)
    >>> print(ghost_ds)
    <xarray.Dataset>
    Dimensions:                 (station: 13, htime: 2160)
    Coordinates:
      * station                 (station) object 'STA-AD0944A_UVP' ... 'STA_ES208...
      * htime                   (htime) datetime64[ns] 2018-01-01 ... 2018-03-31T...
    Data variables:
        h_sconco3               (station, htime) float64 45.61 45.11 ... 52.12 51.12
        longitude               (station) float64 1.565 -7.302 ... -6.193
        latitude                (station) float64 42.52 40.23 ... 38.08 43.07
        altitude                (station) float64 1.637e+03 ... 1.253e+03
        station_classification  (station) object 'background' ... 'backgro...
        area_classification     (station) object 'rural' 'rural' ... 'rural'
        country                 (station) object 'Andorra' ... 'Spain'
    """

    # Load librairies
    import numpy as np
    import pandas as pd
    import copy
    import xarray as xr

    # Check type of inputs
    assert GetType(htime) in ['pandas.core.indexes.datetimes.DatetimeIndex',
                              'pd.core.indexes.datetimes.DatetimeIndex'],'xtime should be a pandas datetime index'
    assert GetType(stations_info)=='pandas.core.frame.DataFrame','stations_info should be a pandas dataframe'
    assert GetType(pollutant)=='str','pollutant should be a string'
    assert GetType(path_ghost)=='str','path_ghost should be a string'
    assert GetType(drop_qa) in ['NoneType','list'],'drop_qa should be a list (or None)'
    if GetType(drop_qa)=='list':
        assert GetType(drop_qa[0])=='int','drop_qa components should be integers'
    assert GetType(n_jobs)=='int','n_jobs should be an integer'
    assert n_jobs > 0,'n_jobs should be strictly positive' 
    assert GetType(verbose)=='bool','verbose should be a boolean'
                
    # Get list of month in YYYYMM format
    list_yyyymm = np.unique(htime.strftime('%Y%m'))

    # Select stations
    if n_jobs>1:
        #(parallel)
        import multiprocessing
        jobargs = [(yyyymm,
                    htime,
                    stations_info,
                    pollutant,
                    path_ghost,
                    drop_qa,
                    timescale,
                    verbose) for iyyyymm, yyyymm in enumerate(list_yyyymm)]
        pool = multiprocessing.Pool(n_jobs)
        allres = pool.starmap(GetGHOSTDataMonth, jobargs)
        pool.close()
        pool.join()
        for iyyyymm,res in enumerate(allres):
            print(iyyyymm)
            output = copy.deepcopy(res) if iyyyymm==0 else xr.concat([output,res],dim='htime',data_vars='minimal')
            if len(output['h_{}'.format(pollutant)].attrs.keys())==0 and len(res['h_{}'.format(pollutant)].attrs.keys())!=0:
                output['h_{}'.format(pollutant)].attrs = res['h_{}'.format(pollutant)].attrs
            print(iyyyymm)   
    else:
        #(sequential)
        for iyyyymm, yyyymm in enumerate(list_yyyymm):
            res = GetGHOSTDataMonth(yyyymm,
                                    htime=htime,
                                    stations_info=stations_info,
                                    pollutant=pollutant,
                                    path_ghost=path_ghost,
                                    drop_qa=drop_qa,
                                    timescale=timescale,
                                    verbose=verbose)
            #return(res)
            #if timescale==['hourly']:
            output = copy.deepcopy(res) if iyyyymm==0 else xr.concat([output,res],dim='htime',data_vars='minimal')
            if len(output['h_{}'.format(pollutant)].attrs.keys())==0 and len(res['h_{}'.format(pollutant)].attrs.keys())!=0:
                output['h_{}'.format(pollutant)].attrs = res['h_{}'.format(pollutant)].attrs
            #elif timescale==['daily']:
            #    output = copy.deepcopy(res) if iyyyymm==0 else xr.concat([output,res],dim='dtime',data_vars='minimal')
            #    if len(output['d_{}'.format(pollutant)].attrs.keys())==0 and len(res['d_{}'.format(pollutant)].attrs.keys())!=0:
            #        output['d_{}'.format(pollutant)].attrs = res['d_{}'.format(pollutant)].attrs
                
    return(output)



def GetGHOSTDataMonth(yyyymm,
                      htime=None,
                      stations_info=None,
                      pollutant=None,
                      path_ghost=None,
                      drop_qa=None,
                      timescale=['hourly'],
                      verbose=False):                      
    """ Sub-function of GetGHOSTData for getting GHOST concentration data, focusing
    on a specific month (for parallelisation).

    See documentation of GetGHOSTData for more detail information.

    Parameters (additional)
    ----------

    yyyymm : str
        Year and month (in YYYYMM format)

    """
    # Load libraries
    import time
    start = time.time()
    import numpy as np
    import pandas as pd
    import xarray as xr
    import warnings
    import os
    import copy
    import timezonefinder
    import pytz
    tf = timezonefinder.TimezoneFinder()

    from datetime import datetime,timedelta
    #istep =0
    #step0 = time.time()
    #print('{} (duration:{})'.format(istep,str(timedelta(seconds=time.time()-step0))))  ; istep = istep+1 ; step0 = time.time() 

    # Check type of inputs
    assert GetType(yyyymm)=='str','yyyymm should be a string'
    assert GetType(htime) in ['pandas.core.indexes.datetimes.DatetimeIndex',
                              'pd.core.indexes.datetimes.DatetimeIndex'],'xtime should be a pandas datetime index'
    assert GetType(stations_info) in ['pandas.core.frame.DataFrame',
                                      'pd.core.frame.DataFrame'],'stations_info should be a pandas dataframe'
    assert GetType(pollutant)=='str','pollutant should be a string'
    assert GetType(path_ghost)=='str','path_ghost should be a string'
    assert GetType(drop_qa) in ['NoneType','list'],'drop_qa should be a list (or None)'
    if GetType(drop_qa)=='list':
        assert GetType(drop_qa[0])=='int','drop_qa components should be integers'
    assert GetType(verbose)=='bool','verbose should be a boolean'

    # Define the file of the GHOST monthly netcdf files
    if timescale==['hourly']:
        ncfile = '{}/hourly/{}/{}_{}.nc'.format(path_ghost,
                                                pollutant,
                                                pollutant,
                                                yyyymm)
    elif timescale==['daily']:
        ncfile = '{}/daily/{}/{}_{}.nc'.format(path_ghost,
                                               pollutant,
                                               pollutant,
                                               yyyymm)
        
    # Define concentration arrays
    stations = stations_info['station'].values
    whtime_yyyymm = [ix for ix,x in enumerate(htime) if x.strftime('%Y%m')==yyyymm]
    conc_noqa = np.full([len(stations),len(whtime_yyyymm)], np.nan)
    conc_qa = np.full([len(stations),len(whtime_yyyymm)], np.nan)
        
    # Define empty array of local time ( bug in local_time)
    missing_date = pd.to_datetime(datetime.strptime('1900010100','%Y%m%d%H'))
    htime_local = np.full([len(stations),len(whtime_yyyymm)], missing_date, dtype='datetime64[ns]')
    #default_offset = timedelta(0)
    #utc_offset = np.full([len(stations)],default_offset,dtype='timedelta64[ns]')
    #utc_offset_hours = np.full([len(stations)],np.nan)
    #print('{} (duration:{})'.format(istep,str(timedelta(seconds=time.time()-step0))))  ; istep = istep+1 ; step0 = time.time()  
    #for istat in range(stations_info.shape[0]):
    #    timezone_str = tf.certain_timezone_at(lat = stations_info.iloc[0]['station_latitude'],
    #                                          lng = stations_info.iloc[0]['station_longitude'])   
    #    timezone = pytz.timezone(timezone_str)
    #    tmp = timezone.utcoffset(datetime.utcnow())   #(datetime timedelta)
    #    utc_offset[istat] = copy.deepcopy(tmp)                 #(numpy timedelta)
    #    utc_offset_hours[istat] = {'-1':-1, '0':+1}[str(tmp.days)]*tmp.seconds/3600  
    
    # Fill concentration arrays
    if os.path.exists(ncfile)==True:
        # Read dataset             
        ds = xr.open_dataset(ncfile)
        ds.close()      
        
        # Select concentrations at requested stations
        wstat = np.intersect1d(ds.station_reference.values,np.array(stations),return_indices=True)
        conc_noqa[wstat[2],:] = ds[pollutant][dict(station=wstat[1])].values
        conc_qa = copy.deepcopy(conc_noqa)
        
            # Quality-Assurance (QA) filtering (!!! need to use "values" to ensure that the changes are passed to "conc_qa" !!!)
        if not drop_qa is None:
            w = np.isin(ds.qa[dict(station=wstat[1])].values,drop_qa).any(axis=2)
            ntrue, nfalse = len(np.where(w.flatten()==True)[0]), len(np.where(w.flatten()==False)[0])
            values = conc_qa[wstat[2],:]
            values[w] = np.nan
            conc_qa[wstat[2],:] = values
            print('QA: Removing {}({:0.1f}%) MEAN/MIN/MAX={:0.1f}/{:0.1f}/{:0.1f} ==> {:0.1f}/{:0.1f}/{:0.1f}'.\
                  format(ntrue,
                         ntrue/(ntrue+nfalse)*100,
                         np.nanmean(conc_noqa),np.nanmin(conc_noqa),np.nanmax(conc_noqa),
                         np.nanmean(conc_qa),np.nanmin(conc_qa),np.nanmax(conc_qa)))
                                     

        # Get local time at selected stations (bug in GHOST)
        if timescale==['hourly']:  
            htime_local[wstat[2],:] = ds.local_time[dict(station=wstat[1])].values

    # Build output dataframe
    if timescale==['hourly']:  
        output = xr.Dataset({'h_{}'.format(pollutant) : (['station','htime'], conc_qa),
                             'htime_local'            : (['station','htime'], htime_local)},
                            coords = {'htime':htime[whtime_yyyymm],                                  
                                      'station':stations})
    elif timescale==['daily']:
        output = xr.Dataset({'h_{}'.format(pollutant) : (['station','htime'], conc_qa)},
                            coords = {'htime':htime[whtime_yyyymm],                                  
                                      'station':stations})
        
    # Add units attribute
    if 'ds' in locals():
        output['h_{}'.format(pollutant)].attrs = ds[pollutant].attrs
            
    # If the user sent more information on stations, include them in the output dataset
    if GetType(stations_info) in ['pandas.core.frame.DataFrame','pd.core.frame.DataFrame']: 
        ds_stations_info = xr.Dataset.from_dataframe(stations_info)
        ds_stations_info.coords['index'] = ds_stations_info.station
        ds_stations_info = ds_stations_info.drop('station').rename({'index':'station'})
        output = xr.merge([output,ds_stations_info])

    print('{} (duration:{})'.format(yyyymm,str(timedelta(seconds=time.time()-start))))

    return(output)    


def GetLonLatGridFromReducedCMAQ(ds):
    """ Get the lon-lat grid from a CMAQ reduced netcdf file (in rotated projection).

    Parameters
    ----------
    ds : xarray dataset
        data set from CMAQ reduced netcdf file

    Output
    ------

    output : xarray dataset with
       - lon : longitude of the center of the cell
       - lat : latitude of the center of the cell
       - lon_b : longitude of the bounds (corners) of the cell
       - lat_b : latitude of the bounds (corners) of the cell    

    Example
    -------
    >>> import xarray as xr
    >>> from pyeslib import pyeslib
    >>> fn = ''
    >>> ds = xr.open_dataset(fn)
    >>> ds.close()
    >>> grid_cmaq = pyeslib.GetLonLatGridFromReducedCMAQ(ds)
    >>> print(grid_cmaq)
    """
    # Load libraries
    import xarray as xr
    import numpy as np
    import cartopy.crs as ccrs
    from datetime import datetime

    # Check type of inputs
    assert GetType(ds) in ['xarray.core.dataset.Dataset','xr.core.dataset.Dataset'],'ds should be an xarray dataset'
    
    # Read longitude and latitude of grid center
    lon = ds.lon.values
    lat = ds.lat.values

    # Get number of cells in longitude and latitude directions
    nx,ny = lon.shape[1],lon.shape[0]

    # Define lon/lat bounds arrays
    lon_b = np.full([ny+1,nx+1],np.nan)
    lat_b = np.full([ny+1,nx+1],np.nan)

    # Get gridded lon/lat resolution
    res_lon = np.full([ny,nx],np.nan)
    res_lon[:,:-1] = lon[:,1:]-lon[:,:-1]
    res_lon[:,-1] = res_lon[:,-2]
    res_lat = np.full([ny,nx],np.nan)
    res_lat[:-1,:] = lat[1:,:]-lat[:-1,:]
    res_lat[-1,:] = res_lat[-2,:]

    # Compute longitude bounds
    lon_b[:-1,:-1] = lon - res_lon             #(all except last column and last line)
    lon_b[:-1,-1] = lon[:,-1] + res_lon[:,-1]  #(last column without last line)
    lon_b[-1,:] = lon_b[-2,:]                  #(last line)

    # Compute latitude bounds
    lat_b[:-1,:-1] = lat - res_lat             #(all except last column and last line)
    lat_b[-1,:-1] = lat[-1,:] + res_lat[-1,:]  #(last line with last column)
    lat_b[:,-1] = lat_b[:,-2]                  #(last column)
    
    # Create a grid xarray dataset
    output = xr.Dataset({'lon': (['y','x'], lon),
                         'lat': (['y','x'], lat),
                         'lon_b': (['y_b','x_b'], lon_b),
                         'lat_b': (['y_b','x_b'], lat_b)})

    # Add basic attributes
    output.attrs.update({'grid_type'     : 'longitude-latitude curvilinear grid from CMAQ',
                         'creation_date' : str(datetime.today())})

    return(output)


                                                         
def GetLonLatGridFromReducedMONARCH(ds):
    """ Get the lon-lat grid from a MONARCH reduced netcdf file (in rotated projection).

    Parameters
    ----------
    ds : xarray dataset
        data set from MONARCH reduced netcdf file

    Output
    ------

    output : xarray dataset with
       - lon : longitude of the center of the cell
       - lat : latitude of the center of the cell
       - lon_b : longitude of the bounds (corners) of the cell
       - lat_b : latitude of the bounds (corners) of the cell    

    Example
    -------
    >>> import xarray as xr
    >>> from pyeslib import pyeslib
    >>> fn = '/esarchive/exp/monarch/a426/d02/hourly/sconco3/sconco3-000_2019080100.nc'
    >>> ds = xr.open_dataset(fn)
    >>> ds.close()
    >>> grid_monarch = pyeslib.GetLonLatGridFromReducedMONARCH(ds)
    >>> print(grid_monarch)
         <xarray.Dataset>
         Dimensions:  (y: 301, x: 301, y_b: 302, x_b: 302)
         Dimensions without coordinates: y, x, y_b, x_b
         Data variables:
             lon      (y, x) float64 -9.873 -9.82 -9.766 -9.713 ... 5.642 5.717 5.791
             lat      (y, x) float64 31.0 31.01 31.03 31.04 ... 48.31 48.31 48.32 48.32
             lon_b    (y_b, x_b) float64 -9.91 -9.857 -9.803 -9.75 ... 5.721 5.796 5.791
             lat_b    (y_b, x_b) float64 30.94 30.95 30.96 30.98 ... 48.27 48.27 48.32
    """
    # Load libraries
    import xarray as xr
    import numpy as np
    import cartopy.crs as ccrs
    from datetime import datetime

    # Check type of inputs
    assert GetType(ds) in ['xarray.core.dataset.Dataset','xr.core.dataset.Dataset'],'ds should be an xarray dataset'
    
    # Read longitude and latitude of grid center
    lon = ds.lon.values
    lat = ds.lat.values

    # Get number of cells in longitude and latitude directions
    nx,ny = lon.shape[1],lon.shape[0]

    # Create cartopy coordinate reference system for the specific type non-standard grid, on WGS84 ellipsoid
    pole_longitude = np.float32(ds.rotated_pole.grid_north_pole_longitude)
    pole_latitude = np.float32(ds.rotated_pole.grid_north_pole_latitude)
    non_regular_grid_crs = ccrs.RotatedPole(pole_longitude=pole_longitude, pole_latitude=pole_latitude)

    # Define a regular gridded coordinate reference system (Plate Carree), on WGS84 ellipsoid
    plate_carree_crs = ccrs.PlateCarree()

    # Convert center lon-lat to coordinates in rotated projection
    res = non_regular_grid_crs.transform_points(plate_carree_crs,ds.lon.values,ds.lat.values)
    rlon = res[:,:,0]
    rlat = res[:,:,1]
    
    # Get resolution in rotated projection
    rlon_res = np.abs(np.nanmean(rlon[:,1:] - rlon[:,:-1]))
    rlat_res = np.abs(np.nanmean(rlat[1:,:] - rlat[:-1,:]))

    # Get grid bounds in rotated projection
    rlon_b = np.full([ny+1,nx+1],np.nan)
    rlat_b = np.full([ny+1,nx+1],np.nan)
    rlon_b[:-1,:-1] = rlon - rlon_res
    rlon_b[-1,:] = rlon_b[-2,:]
    rlon_b[:,-1] = rlon_b[:,-2]+rlon_res
    rlat_b[:-1,:-1] = rlat - rlat_res
    rlat_b[-1,:] = rlat_b[-2,:]+rlat_res 
    rlat_b[:,-1] = rlat_b[:,-2]

    # check some cells
    def CellInfo(y,x):
        nw_corner = 'NW: ({:12.6f};{:12.6f})'.format(rlat_b[y+1,x],rlon_b[y+1,x])
        ne_corner = 'NE: ({:12.6f};{:12.6f})'.format(rlat_b[y+1,x+1],rlon_b[y+1,x+1])
        sw_corner = 'SW: ({:12.6f};{:12.6f})'.format(rlat_b[y,x],rlon_b[y,x])
        se_corner = 'SE: ({:12.6f};{:12.6f})'.format(rlat_b[y,x+1],rlon_b[y,x+1])
        print('{}         {}'.format(nw_corner,ne_corner))                                                                                       
        print('{}         {}'.format(sw_corner,se_corner))                                                                                       
    #CellInfo(129,0)
    #CellInfo(130,0)
        
    # Convert grid bounds from rotated projection to lon-lat
    res = plate_carree_crs.transform_points(non_regular_grid_crs,rlon_b,rlat_b)
    lon_b = res[:,:,0]
    lat_b = res[:,:,1]

    # Create a grid xarray dataset
    output = xr.Dataset({'lon': (['y','x'], lon),
                         'lat': (['y','x'], lat),
                         'lon_b': (['y_b','x_b'], lon_b),
                         'lat_b': (['y_b','x_b'], lat_b)})

    # Add basic attributes
    output.attrs.update({'grid_type'     : 'longitude-latitude non regular (rotated) grid from MONARCH',
                         'creation_date' : str(datetime.today())})

    return(output)


def GetType(x):
    """ Return the type of the variable in string format
    """
    return(str(type(x)).split("'")[1])


def LocateStationsOnGrid(stations_longitude,
                         stations_latitude,
                         grid,
                         verbose=False):
    """ Locate in which grid cells are located a given set of stations.

    Parameters
    ----------
    stations_longitude : array of float
        Longitude of the stations
    
    stations_longitude : array of float
        Longitude of the stations
    
    grid : xarray dataset 
        Grid dataset with lon, lat, lon_b(ounds), lat_b(ounds) information

    verbose : bool, default=False
        To print intermediate information

    Output
    ------
    dictionnary with :
        - stations_ix (list of integers) : X-axis index of the stations on the grid
        - stations_iy (list of integers) : Y-axis index of the stations on the grid
        - ncells (integer) : number of cells with at least one station
        - cells_ix (list of integers) : X-axis index of the cells with at least one station
        - cells_iy (list of integers) : Y-axis index of the cells with at least one station
        - cells_nstations (list of integers) : number of station in the cells with at least one station 

    Example
    -------
    >>> import numpy as np
    >>> from pyeslib import pyeslib
    >>> 
    >>> extent,reslon,reslat = [-10., 5., 35., 45.], 1, 1
    >>> grid = pyeslib.DefineLonLatRegularGrid(extent, reslon, reslat)
    >>> 
    >>> stations_lon = np.array([ 1.56525 , -7.301944, -8.466111])
    >>> stations_lat = np.array([42.516944, 40.233056, 39.3525  ])
    >>> location = pyeslib.LocateStationsOnGrid(stations_lon, stations_lat, grid)
    >>> print(location_dict)
    {'stations_ix': array([11,  2,  1]), 'stations_iy': array([7, 5, 4]), 'ncells': 3, 'cells_ix': array([11,  2,  1]), 'cells_iy': array([7, 5, 4]), 'cells_nstations': array([1, 1, 1])}
    ([11, 2, 1], [7, 5, 4])
    >>> 
    >>> stations_lat = np.array([42.516944, 40.233056, 90])
    >>> location_dict = pyeslib.LocateStationsOnGrid(stations_lon, stations_lat, grid)
    INFO: only 2/3 (66.7%) correctly located on the grid
    >>> print(location_dict)
    {'stations_ix': array([11.,  2., nan]), 'stations_iy': array([ 7.,  5., nan]), 'ncells': 2, 'cells_ix': array([11,  2]), 'cells_iy': array([7, 5]), 'cells_nstations': array([1, 1])}    
    """
    # Load libraries
    import numpy as np
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import xarray as xr

    # Check type of inputs
    assert GetType(stations_longitude) in ['numpy.ndarray','np.ndarray'],'stations_longitude should be a numpy array'
    assert GetType(stations_latitude) in ['numpy.ndarray','np.ndarray'],'stations_latitude should be a numpy array'
    assert GetType(grid) in ['xarray.core.dataset.Dataset','xr.core.dataset.Dataset'],'grid should be an xarray dataset'
    assert GetType(verbose)=='bool','verbose should a boolean'
        
    # Locate stations on the grid
    stations_ix,stations_iy = [],[]
    for istat in range(len(stations_longitude)):
        
        # Initialize
        iy,ix = np.nan,np.nan

        # Make a first guess on the location of the station
        dist = np.abs(grid.lon.values-stations_longitude[istat])+np.abs(grid.lat.values-stations_latitude[istat])
        w = np.where(dist==np.nanmin(dist))
        
        # Check if this first guess is indeed located within the corresponding grid cell polygon
        located = False
        if (len(w[0]),len(w[1]))==(1,1):
            iy_tmp,ix_tmp = w[0][0],w[1][0]

            polygon = Polygon(np.column_stack(
                (np.array([float(grid.lon_b.sel(x_b=ix_tmp  , y_b=iy_tmp  )),
                           float(grid.lon_b.sel(x_b=ix_tmp  , y_b=iy_tmp+1)),
                           float(grid.lon_b.sel(x_b=ix_tmp+1, y_b=iy_tmp+1)),
                           float(grid.lon_b.sel(x_b=ix_tmp+1, y_b=iy_tmp  ))]),
                 np.array([float(grid.lat_b.sel(x_b=ix_tmp  , y_b=iy_tmp  )),
                           float(grid.lat_b.sel(x_b=ix_tmp  , y_b=iy_tmp+1)),
                           float(grid.lat_b.sel(x_b=ix_tmp+1, y_b=iy_tmp+1)),
                           float(grid.lat_b.sel(x_b=ix_tmp+1, y_b=iy_tmp  ))]))))
            point = Point(stations_longitude[istat], stations_latitude[istat])
            # check if located inside cell
            if point.within(polygon):
                iy,ix = iy_tmp,ix_tmp
                located = True
            else:
                # if not, search in surrounding cells
                for iy_tmp in np.arange(w[0][0]-1,w[0][0]+2,1):
                    for ix_tmp in np.arange(w[1][0]-1,w[1][0]+2,1):

                        if iy_tmp < 0 or iy_tmp >= grid.dims['y'] or ix_tmp < 0 or ix_tmp >= grid.dims['x']: continue
                                                
                        if located==False:
                            polygon = Polygon(np.column_stack(
                                (np.array([float(grid.lon_b.sel(x_b=ix_tmp  , y_b=iy_tmp  )),
                                           float(grid.lon_b.sel(x_b=ix_tmp  , y_b=iy_tmp+1)),
                                           float(grid.lon_b.sel(x_b=ix_tmp+1, y_b=iy_tmp+1)),
                                           float(grid.lon_b.sel(x_b=ix_tmp+1, y_b=iy_tmp  ))]),
                                 np.array([float(grid.lat_b.sel(x_b=ix_tmp  , y_b=iy_tmp  )),
                                           float(grid.lat_b.sel(x_b=ix_tmp  , y_b=iy_tmp+1)),
                                           float(grid.lat_b.sel(x_b=ix_tmp+1, y_b=iy_tmp+1)),
                                           float(grid.lat_b.sel(x_b=ix_tmp+1, y_b=iy_tmp  ))]))))
                            point = Point(stations_longitude[istat], stations_latitude[istat])
                            if point.within(polygon):
                                iy,ix = iy_tmp,ix_tmp
                                located = True

                
            

        if located==False:
            print('*** NOT LOCATED *** istat={} ; lat,lon={:0.6f},{:0.6f}  ({},{})'.\
                  format(istat,stations_latitude[istat],stations_longitude[istat],w[0][0],w[1][0]))
        
        # Append the result
        stations_iy.append(iy)
        stations_ix.append(ix)

            
    # Print information message if some stations not located on the grid
    if verbose:
        nstations = len(stations_ix)
        nstations_correctly_located = len(np.where(np.isfinite(np.array(stations_ix)))[0])
        if nstations!=nstations_correctly_located:
            print('INFO: only {}/{} ({:0.1f}%) correctly located on the grid'.format(nstations_correctly_located,nstations,nstations_correctly_located/nstations*100))

    # Get more information
    wstat = np.where((np.isfinite(stations_ix)) & (np.isfinite(stations_iy)))[0]
    ixiy = np.array(['{}_{}'.format(stations_ix[istat],stations_iy[istat]) for istat in wstat])    
    ncells = len(np.unique(ixiy))
    ixiy_unique = np.unique(ixiy)
    cells_ix = [int(x.split('_')[0]) for x in ixiy_unique] 
    cells_iy = [int(x.split('_')[1]) for x in ixiy_unique]
    cells_nstations = [len(np.where((np.array(stations_ix)==int(x.split('_')[0])) \
                                    & (np.array(stations_iy)==int(x.split('_')[1])))[0]) \
                       for x in np.unique(ixiy)]

    output = {'stations_ix'     : np.array(stations_ix),
              'stations_iy'     : np.array(stations_iy),
              'ncells'          : ncells,
              'cells_ix'        : np.array(cells_ix),
              'cells_iy'        : np.array(cells_iy),
              'cells_nstations' : np.array(cells_nstations)}
    
    return(output)



def MemoryUsage(verbose=True):
    """ Provide information on the use of memory, inside the 
    python environment and on the node

    Parameters
    ----------

    verbose : bool, default=False
        To print intermediate information   

    Outputs
    -------
    output : dict
        Dictionnary with information on memory usage (in GB):
         - memory_node_used: memory currently used on the node
         - memory_node_free: memory currently free on the node
         - memory_node_total: total memory of the node
         - percentage_node_used: percentage of memory used on the node
         - percentage_node_free: percentage of memory free on the node 
         - memory_python_used: memory currently used by python variables    
    
    """
    # Load libraries
    import os
    import sys

    # Read top header
    tmp = os.popen('top -b | head -4 | grep "Mem"').read()

    # Extract memory information
    memory_node_total = float(tmp.split(',')[0].split('total')[0].split(':')[1])
    memory_node_free = float(tmp.split(',')[1].split('free')[0])
    memory_node_used = float(tmp.split(',')[2].split('used')[0])

    # Get unit factor to convert to GB
    unit_factor = {'KiB':1/1e6, 'MiB':1/1e3}[tmp[:3]]

    # Convert memory information to GB
    memory_node_total = memory_node_total * unit_factor
    memory_node_free  = memory_node_free * unit_factor
    memory_node_used  = memory_node_used * unit_factor

    # Get all memory used (NOT SURE WHAT MEANS THE RESULTS HERE?...)
    #memory_python_used = 0.
    #for var,obj in list(globals().items()):
    #    try:
    #        memory_python_used += sys.getsizeof(obj)/1e3 #(MB -> GB)
    #    except:
    #        do_nothing = True

    # Print results
    if verbose:
        print('Memory used on the node           = {:0.3f}/{:0.3f} GB ({:0.2f}%)'.\
              format(memory_node_used,memory_node_total,memory_node_used/memory_node_total*100))
        print('Memory free on the node           = {:0.3f}/{:0.3f} GB ({:0.2f}%)'.\
              format(memory_node_free,memory_node_total,memory_node_free/memory_node_total*100))
        #print('Memory used with python variables = {:0.3f} GB'.\
        #      format(memory_python_used))

    # Create output dictionnary
    output = {'memory_node_used':memory_node_used,
              'memory_node_free':memory_node_free,
              'memory_node_total':memory_node_total,
              'percentage_node_used':memory_node_used/memory_node_total*100,
              'percentage_node_free':memory_node_free/memory_node_total*100}
    #'memory_python_used':memory_python_used}
    return(output)
    

def GetRegionInformationOnGrid(grid,
                               nuts='EUR',
                               countries_focus=None,
                               nrank=100,
                               n_jobs=1):
    """ Get information on the NUTS regions covering the cells of a grid.

    GetRegionInformationOnGrid extracts geographical information on each grid cell 
    from shapefiles. In each cell, it identifies the different overlapping (country/NUTS)
    regions, and extract information on the area, areas of intersection, in absolute and 
    percentage. Different regional products are available, inculding European countries, 
    and NUTS-0 (country-scale) or NUTS-1 / NUTS-2 / NUTS-3 (finer and finer administative areas)
    or FUA (functional urban areas of major European cities).

    Parameters
    ----------
    grid : xarray dataset 
        Grid dataset with lon/lat/lon_b/lat_b variables, and x/y dimensions.
    
    nuts : string, default="EUR-country"
        Type of regional information requested :
           - "EUR"    : English-names World countries
           - "WORLD"  : French-names World countries
           - "NUTS-0" : NUTS-0 regions (equivalent to countries)
           - "NUTS-1" : NUTS-1 regions
           - "NUTS-2" : NUTS-2 regions (equivalent to CCAA in Spain)
           - "NUTS-3" : NUTS-2 regions (equivalent to provinces in Spain)
           - "FUA"    : FUA (main European cities)
    
    countries_focus : list of country code (2 capital letters) strings
        List of countries where to search information. If it is omitted, the function will
        search over all regions available in the shapefile (much slower). By specifying a 
        list of countries of interest, that overlap (or not) the grid, it makes the code 
        much faster.

    n_jobs : int, default=1
        Number of jobs to run in parallel. If n_jobs > 1, then the function will
        run in parallel, each month being run in parallel on a specific job.

    nrank : int, default=100
        The regions identified in each grid cell are ranked by the percentage of the 
        grid cell covered by the region (percentage_area_intersect_cell). This parameter
        indicates the maximum number of regions identified in each cell (it is one of the
        dimension of the output dataset).

    Output
    ------

    output : xarray dataset with :
        - country : country (2 letters) code
        - icountry : index of the country (in list_country dimension)
        - region : NUTS region
        - iregion : index of the NUTS region (in list_region dimension)
        - percentage_area_intersect_cell : percentage of the grid cell covered by the region (in %)
        - percentage_area_intersect_region : percentage of the region covered by the grid cell (in %)
        - area_intersect : area of intersection between grid cell and region (in km2)
        - area_cell : area of the grid cell (in km2)
        - area_region : area of the region (in km2)
        - list_country : list of all individual country (2 letters) codes
        - list_region : list of all individual regions
    The rank (set to 100)

    Example
    -------

    >>> from pyeslib import pyeslib
    >>>
    >>> # Get a 1°x1° grid dataset over a domain
    >>> extent,reslon,reslat = [-10., 5., 35., 45.], 1, 1
    >>> grid = pyeslib.DefineLonLatRegularGrid(extent, reslon, reslat)  
    >>> 
    >>> # Extract different regional information (and one specific cell for the example)
    >>> for nuts in ['EUR','NUTS-0','NUTS-1','NUTS-2','NUTS-3']:
    >>>     ds = pyeslib.GetRegionInformationOnGrid(grid,nuts=nuts,countries_focus=['ES','FR','AD'],n_jobs=1)
    >>>     ix,iy = 11,7
    >>>     print('\n {} regions covering the grid cell ({},{}) centered in lon/lat={:0.1f}/{:0.1f}'.\
    >>>           format(nuts,ix,iy,grid.lon.isel(x=ix,y=iy).values,grid.lat.isel(x=ix,y=iy).values))
    >>>     for irank in range(ds.dims['rank']):
    >>>         region                           = ds.region.isel(x=ix,y=iy,rank=irank).values
    >>>         country                          = ds.country.isel(x=ix,y=iy,rank=irank).values
    >>>         percentage_area_intersect_cell   = ds.percentage_area_intersect_cell.isel(x=ix,y=iy,rank=irank).values
    >>>         percentage_area_intersect_region = ds.percentage_area_intersect_region.isel(x=ix,y=iy,rank=irank).values
    >>>         if region!='':
    >>>             print('{:3d}) region "{:25s}" ({}) {:5.1f}% of the cell    ({:5.1f}% of the region)'.\
    >>>                   format(irank, region, country, percentage_area_intersect_cell, percentage_area_intersect_region))

        EUR regions covering the grid cell (11,7) centered in lon/lat=1.5/42.5
         0) region "Spain                    " (ES)  59.5% of the cell    (  1.1% of the region)
         1) region "France                   " (FR)  37.7% of the cell    (  0.5% of the region)
         2) region "Andorra                  " (AD)   2.9% of the cell    (100.0% of the region)

        NUTS-0 regions covering the grid cell (11,7) centered in lon/lat=1.5/42.5
         0) region "España                   " (ES)  59.5% of the cell    (  1.1% of the region)
         1) region "France                   " (FR)  37.7% of the cell    (  0.5% of the region)

        NUTS-1 regions covering the grid cell (11,7) centered in lon/lat=1.5/42.5
         0) region "Este                     " (ES)  59.5% of the cell    (  8.9% of the region)
         1) region "Occitanie                " (FR)  37.7% of the cell    (  4.7% of the region)

        NUTS-2 regions covering the grid cell (11,7) centered in lon/lat=1.5/42.5
         0) region "Cataluña                 " (ES)  59.5% of the cell    ( 16.5% of the region)
         1) region "Midi-Pyrénées            " (FR)  34.4% of the cell    (  6.8% of the region)
         2) region "Languedoc-Roussillon     " (FR)   3.3% of the cell    (  1.1% of the region)

        NUTS-3 regions covering the grid cell (11,7) centered in lon/lat=1.5/42.5
         0) region "Lleida                   " (ES)  47.9% of the cell    ( 32.9% of the region)
         1) region "Ariège                   " (FR)  34.4% of the cell    ( 62.5% of the region)
         2) region "Barcelona                " (ES)   6.9% of the cell    (  8.9% of the region)
         3) region "Girona                   " (ES)   4.7% of the cell    (  6.8% of the region)
         4) region "Pyrénées-Orientales      " (FR)   2.7% of the cell    (  8.2% of the region)
         5) region "Aude                     " (FR)   0.6% of the cell    (  0.7% of the region)
    >>> # You can see the difference between EUR where country names are in English and include Andorra,
    >>> # while NUTS-0 correspond to non-English names of the country part of the NUTS classification (which
    >>> # excludes Andorra for instance).
    """
    #Load libraries
    import copy
    import multiprocessing
    import xarray as xr
    import numpy as np

    # Check type of inputs
    assert GetType(grid)=='xarray.core.dataset.Dataset','grid should be an xarray dataset'
    assert GetType(nuts)=='str','nuts should be a string'
    assert GetType(countries_focus) in ['NoneType','list'],'countries_focus should be a list of None'
    assert GetType(nrank)=='int','nrank should be an integer'
    assert GetType(n_jobs)=='int','n_jobs should be an integer'
    assert n_jobs > 0,'n_jobs should be strictly positive'
    
    if n_jobs==1:
        #(sequential)
        for ix in range(grid.dims['x']):
            res = GetRegionInformationOnGridIx(ix,grid,nuts,countries_focus,nrank)
            output = copy.deepcopy(res) if ix==0 else xr.concat([output,res],dim='x',data_vars='minimal')

    elif n_jobs>1:
        #(parallel)
        jobargs = [(ix,grid,nuts,countries_focus,nrank) for ix in range(grid.dims['x'])]
        pool = multiprocessing.Pool(n_jobs)
        allres = pool.starmap(GetRegionInformationOnGridIx, jobargs)
        pool.close()
        pool.join()
        for ix,res in enumerate(allres):
            output = copy.deepcopy(res) if ix==0 else xr.concat([output,res],dim='x',data_vars='minimal')

    # Reduce the rank dimension accordingly to the maximum number of regions identified over the grid cells
    nrank_effective = len(np.where(np.isfinite(output.iregion.mean(['y','x']).values))[0])
    output = output[dict(rank=np.arange(nrank_effective))]
                              
    return(output)




def GetRegionInformationOnGridIx(ix,
                                 grid,
                                 nuts,
                                 countries_focus,
                                 nrank):
    """ Sub-function of GetRegionInformationOnGrid for getting information on 
    the NUTS regions covering the cells of a grid, focusing on the grid cells
    of X index "ix" (for parallelisation).

    See documentation of GetRegionInformationOnGrid for more detail information.
    """
    verbose = False

    print(ix)
    
    # Load libraries
    import numpy as np
    import geopandas as gpd
    import pandas as pd
    import shapely.geometry
    import pyproj
    import xarray as xr

    import warnings
    #from shapely.errors import ShapelyDeprecationWarning
    #warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
    
    # Check type of inputs
    assert GetType(ix)=='int','ix should be an integer'
    assert GetType(grid)=='xarray.core.dataset.Dataset','grid should be an xarray dataset'
    assert GetType(nuts)=='str','nuts should be a string'
    assert GetType(countries_focus) in ['NoneType','list'],'countries_focus should be a list of None'
    assert GetType(nrank)=='int','nrank should be an integer'
    
    # Define coordinate reference systems (CRS)
    crs_4326 = pyproj.CRS.from_epsg(4326)   #(EPSG:4326, lon/lat coordinates in degrees)
    crs_3035 = pyproj.CRS.from_epsg(3035)   #(EPSG:3035, x/y coordinates in meters)

    # Get geopandas dataframe in EPSG:3035
    if nuts=='EUR':
        resolution = {'high':'01',
                      'medium':'10',
                      'coarse':'60'}['coarse']

        # Read raw shapefile
        fn = '{}/regions_shapefile/ref-countries-2020-{}m/CNTR_RG_{}M_2020_3035.shp'.format(pyeslib_data_dir,resolution,resolution)
        df_region = gpd.GeoDataFrame.from_file(fn)        #(EPSG:3035)
        df_region.columns = ['COUNTRY','X','REGION','X','X','geometry']

        # Convert to lon/lat (for information purpose)
        df_region_lonlat = df_region.to_crs(crs_4326) 
        
    elif nuts=='WORLD':
        # Read raw shapefile 
        fn = '{}/regions_shapefile/world_countries/world-administrative-boundaries.shp'.format(pyeslib_data_dir)
        df_region_lonlat = gpd.GeoDataFrame.from_file(fn)        #(EPSG:4326)
        df_region_lonlat.columns = ['X','X','X','X','X','REGION','COUNTRY','X','geometry']

        # Remove "None" countries
        df_region_lonlat = df_region_lonlat.iloc[np.where(df_region_lonlat['COUNTRY'].values != None)[0]].reset_index()

        # Convert to x/y
        df_region = df_region_lonlat.to_crs(crs_3035)
        
    elif nuts in ['NUTS-0','NUTS-1','NUTS-2','NUTS-3']:
        resolution = {'high':'01',
                      'medium':'10',
                      'coarse':'60'}['coarse']

        # Read raw shapefile
        fn = '{}/regions_shapefile/NUTS_RG_{}M_2021_3035/NUTS_RG_{}M_2021_3035.shx'.\
            format(pyeslib_data_dir, resolution, resolution)
        df_region = gpd.GeoDataFrame.from_file(fn)        #(EPSG:3035)

        # Select NUTS level
        df_region = df_region[df_region['LEVL_CODE']==int(nuts[-1])]

        # Rename useful columns
        df_region.columns = ['X','X','COUNTRY','REGION','X','X','X','X','X','geometry']  

        # Convert to lon/lat (for information purpose)
        df_region_lonlat = df_region.to_crs(crs_4326)

    elif nuts in ['FUA']:
        # Read raw shapefile
        fn = '/esarchive/recon/jrc/fua/original_files/FUA_Boundaries.shp'
        df_region = gpd.GeoDataFrame.from_file(fn)   

        # Rename useful columns
        df_region.columns = ['X','X','REGION','COUNTRY','X','AREA','geometry']
        
        # Convert to lon/lat (for information purpose)
        df_region_lonlat = df_region.to_crs(crs_4326)

        
    # Get regions and countries
    regions = df_region['REGION'].values
    countries = df_region['COUNTRY'].values
    countries_unique = np.unique(countries)
    nregions = len(regions)

    # Identify countries where to focus
    wfocus = np.arange(nregions) if countries_focus is None else np.array([ix for ix,x in enumerate(countries) if x in countries_focus])

    # Create geopandas dataframe of the grid
    list_polygon_cell_lonlat = []
    for iy in range(grid.dims['y']):
        lon_cell = [float(grid.lon_b.sel(x_b=ix  , y_b=iy  )),
                    float(grid.lon_b.sel(x_b=ix  , y_b=iy+1)),
                    float(grid.lon_b.sel(x_b=ix+1, y_b=iy+1)),
                    float(grid.lon_b.sel(x_b=ix+1, y_b=iy  )),
                    float(grid.lon_b.sel(x_b=ix  , y_b=iy  ))]
        lat_cell = [float(grid.lat_b.sel(x_b=ix  , y_b=iy  )),
                    float(grid.lat_b.sel(x_b=ix  , y_b=iy+1)),
                    float(grid.lat_b.sel(x_b=ix+1, y_b=iy+1)),
                    float(grid.lat_b.sel(x_b=ix+1, y_b=iy  )),
                    float(grid.lat_b.sel(x_b=ix  , y_b=iy  ))]
        polygon_cell_lonlat = shapely.geometry.Polygon(zip(lon_cell, lat_cell))
        list_polygon_cell_lonlat.append(polygon_cell_lonlat)

    tmp = pd.DataFrame({'x': [ix for iy in range(grid.dims['y'])],
                        'y': list(np.arange(grid.dims['y']))})
    df_cell_lonlat = gpd.GeoDataFrame(tmp,
                                      crs='epsg:4326',
                                      geometry=list_polygon_cell_lonlat)
    
    # Convert it to the ESPF:3035 x/y projection (the one of the shapefile)
    df_cell = df_cell_lonlat.to_crs(crs_3035)

    # Create output dataset
    complete_info = {}
    output = xr.Dataset({'country'  : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], '',dtype='object')),
                         'icountry' : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan)),
                         'region'   : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], '', dtype='object')),
                         'iregion'  : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan)),
                         'percentage_area_intersect_cell'   : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan)),
                         'percentage_area_intersect_region' : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan)),
                         'area_intersect' : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan)),
                         'area_cell'      : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan)),
                         'area_region'    : (['y','x','rank'], np.full([grid.dims['y'],1,nrank], np.nan))},
                        coords={'list_country' : countries_unique,
                                'list_region'  : regions,
                                'x' : np.array([ix]),
                                'y' : np.arange(grid.dims['y'])})

    # Loop on the grid                                   
    for iy in range(grid.dims['y']):
        #if iy!=6: continue

        polygon_cell = df_cell.iloc[iy]['geometry']
        polygon_cell_lonlat = df_cell_lonlat.iloc[iy]['geometry']
        if verbose: print(polygon_cell_lonlat)        
        info_cell = {'country':[],
                     'icountry':[],
                     'region':[],
                     'iregion':[],
                     'percentage_area_intersect_cell':[],
                     'percentage_area_intersect_region':[],
                     'area_intersect':[],
                     'area_cell':[],
                     'area_region':[]}

        # Loop on the regions
        for ireg in wfocus:
            
            #if regions[ireg] in ['Antarctica','New Zealand','Norfolk Island',
            #                     'Wallis and Futuna','Vanuatu','Tuvalu','Tonga',
            #                     'Nauru','New Caledonia','Solomon Islands',
            #                     'Kiribati','Kiribati','Australia']: continue
            #French Polynesia'
            
            polygon_region = df_region.iloc[ireg]['geometry']
            polygon_region_lonlat = df_region_lonlat.iloc[ireg]['geometry']

            

            intersect_xy = polygon_cell.intersects(polygon_region)
            intersect_lonlat = polygon_cell_lonlat.intersects(polygon_region_lonlat)
            

            if intersect_xy==False and intersect_lonlat==True:
                if verbose: print('--------({}) {}|{}  : issue (lonlat but not xy) skip'.format(iy,ireg,regions[ireg]))
                unclear_issue = True
                
            elif intersect_xy==True and intersect_lonlat==False:
                try:
                    if verbose: print('--------({}) {}|{}  : issue (xy but not lonlat) but still compute'.format(iy,ireg,regions[ireg]))
                    area_intersect = polygon_cell.intersection(polygon_region).area/1e6 #(m2 -> km2)
                    unclear_issue = False
                except:
                    if verbose: print('--------({}) {}|{}  : issue (xy but not lonlat) but still compute but other issue'.format(iy,ireg,regions[ireg]))  
                    unclear_issue = True
                    
            elif intersect_xy==False and intersect_lonlat==False:
                continue
                
            elif intersect_xy==True and intersect_lonlat==True:
                try:
                    if verbose: print('--------({}) {}|{}  : ok'.format(iy,ireg,regions[ireg]))
                    area_intersect = polygon_cell.intersection(polygon_region).area/1e6
                    unclear_issue = False
                except:
                    if verbose: print('--------({}) {}|{}  : ok but finally no'.format(iy,ireg,regions[ireg])) 
                    unclear_issue = True
                    

                
            area_cell = polygon_cell.area/1e6                                   #(m2 -> km2)
            area_region = polygon_region.area/1e6                               #(m2 -> km2) #(NOTE: consistent with AREA given in FAU shapefile)
            if unclear_issue==False:
                percentage_area_intersect_cell = area_intersect/area_cell*100       #(%)
                percentage_area_intersect_region = area_intersect/area_region*100   #(%)  
            else:
                percentage_area_intersect_cell = np.nan
                percentage_area_intersect_region = np.nan
                area_intersect = np.nan
                    
            info_cell['country'].append(countries[ireg])
            info_cell['icountry'].append(np.where(countries_unique==countries[ireg])[0][0])
            info_cell['region'].append(regions[ireg])
            info_cell['iregion'].append(ireg)
            info_cell['percentage_area_intersect_cell'].append(percentage_area_intersect_cell)
            info_cell['percentage_area_intersect_region'].append(percentage_area_intersect_region)
            info_cell['area_intersect'].append(area_intersect)
            info_cell['area_cell'].append(area_cell)
            info_cell['area_region'].append(area_region)
                    
        # Order overlapping regions by decreasing percentage of area of intersect 
        if len(info_cell['percentage_area_intersect_cell'])!=0:
            worder = np.argsort(np.array(info_cell['percentage_area_intersect_cell']))[::-1]
            for irank,iorder in enumerate(worder):
                if irank >= nrank: continue
                for var in list(info_cell.keys()):
                    output[var][dict(x=0,y=iy,rank=irank)] = info_cell[var][iorder]

    return(output)


def PrepareTimeArrays(time1, time2, time_scales=['h','d','m','y']):
    """ Prepare time arrays 

    Parameters 
    ----------       
    time1 : str
        Starting time in format '%Y%m%d%H'.

    time2 : str
        Starting time in format '%Y%m%d%H'.

    time_scales : list of str
        List of time scales to be prepared : "h" for hourly time array, 
        "d" for daily time array, "m" for monthly time array (taken as the first 
        day of the month), "y" for yearly time array (taken as the first day of the year)

    Outputs
    -------

    output : dict
        Dictionnary containing (if the corresponding time scale are requested):
             - time1 : str used as input
             - time2 : str used as input
             - date1 : date corresponding to time1
             - date2 : date corresponding to time2
             - htime : date array at hourly scale
             - nhtime : int with total number of hours
             - dtime : date array at daily scale
             - ndtime : int with total number of days
             - mtime : date array at monthly scale
             - nmtime :int with total number of months
             - ytime : date array at yearly scale
             - nytime : int with total number of years
             - years : numpy arrays with years as integers

    Examples
    --------
    >>> from pyeslib import PrepareTimeArrays
    >>> time1,time2 = '2019010300','2019020523'
    >>> res = PrepareTimeArrays(time1, time2, time_scales=['h','d'])
    >>> print(res.keys())
    dict_keys(['time1', 'time2', 'date1', 'date2', 'htime', 'nhtime', 'dtime', 'ndtime'])
    >>> print(res)
    {'time1': '2019010300', 'time2': '2019020523', 'date1': Timestamp('2019-01-03 00:00:00'), 'date2': Timestamp('2019-02-05 23:00:00'), 'htime': DatetimeIndex(['2019-01-03 00:00:00', '2019-01-03 01:00:00',
               '2019-01-03 02:00:00', '2019-01-03 03:00:00',
               '2019-01-03 04:00:00', '2019-01-03 05:00:00',
               '2019-01-03 06:00:00', '2019-01-03 07:00:00',
               '2019-01-03 08:00:00', '2019-01-03 09:00:00',
               ...
               '2019-02-05 14:00:00', '2019-02-05 15:00:00',
               '2019-02-05 16:00:00', '2019-02-05 17:00:00',
               '2019-02-05 18:00:00', '2019-02-05 19:00:00',
               '2019-02-05 20:00:00', '2019-02-05 21:00:00',
               '2019-02-05 22:00:00', '2019-02-05 23:00:00'],
              dtype='datetime64[ns]', length=816, freq='H'), 'nhtime': 816, 'dtime': DatetimeIndex(['2019-01-03', '2019-01-04', '2019-01-05', '2019-01-06',
               '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10',
               '2019-01-11', '2019-01-12', '2019-01-13', '2019-01-14',
               '2019-01-15', '2019-01-16', '2019-01-17', '2019-01-18',
               '2019-01-19', '2019-01-20', '2019-01-21', '2019-01-22',
               '2019-01-23', '2019-01-24', '2019-01-25', '2019-01-26',
               '2019-01-27', '2019-01-28', '2019-01-29', '2019-01-30',
               '2019-01-31', '2019-02-01', '2019-02-02', '2019-02-03',
               '2019-02-04', '2019-02-05'],
              dtype='datetime64[ns]', freq='D'), 'ndtime': 34}
    """
    # Load libraries
    import datetime
    import pandas as pd
    import numpy as np

    # Check type of inputs
    assert GetType(time1)=='str','time1 should be a string'
    assert GetType(time2)=='str','time2 should be a string'
    assert GetType(time_scales)=='list','time_scales should be a list'
    assert GetType(time_scales[0])=='str','time_scales components should be strings'
    
    # Convert strings into times
    date1 = pd.to_datetime(datetime.datetime.strptime(time1, '%Y%m%d%H')) # (starting time)
    date2 = pd.to_datetime(datetime.datetime.strptime(time2, '%Y%m%d%H')) # (ending time, included)

    # Define first day of the month
    date1_first_day_of_month = pd.to_datetime(datetime.datetime.strptime(time1[0:6]+'01'+time1[8:10], '%Y%m%d%H')) 

    # Define the initial output dictionnary
    output = {'time1' : time1,
              'time2' : time2,
              'date1' : date1,
              'date2' : date2}

    # Add time arrays at hourly scale
    if 'h' in time_scales:
        output['htime'] = pd.date_range(date1, date2, freq='H')       
        output['nhtime'] = len(output['htime'])

    # Add time arrays at daily scale
    if 'd' in time_scales:
        output['dtime'] = pd.date_range(date1, date2, freq='D')         
        output['ndtime'] = len(output['dtime'])

    # Add time arrays at monthly scale (taken as first day of the month)
    if 'm' in time_scales:   
        output['mtime'] = pd.date_range(date1_first_day_of_month, date2, freq='MS') 
        output['nmtime'] = len(output['mtime'])

    # Add time arrays at yearly scale (taken as first day of the year)
    if 'y' in time_scales:
        output['ytime'] = pd.date_range(date1_first_day_of_month, date2, freq='AS') 
        output['nytime'] = len(output['ytime'])
        output['years'] = np.array([i.year for i in output['ytime']])

    return(output)
                


def SelectGHOSTStations(htime,
                        pollutant=None,
                        path_ghost=None,
                        drop_qa=None,
                        extent=None, 
                        station_classification=None, 
                        area_classification=None,
                        n_jobs=1,
                        timescale=['hourly'],
                        verbose=False):
    """ Select GHOST stations over a specific time period.
 
    SelectGHOSTStations applies a set of filter on the geographical 
    extent, station classification, area classification, pollutant, to 
    identify stations available for a given network and a given GHOST version,
    eventually applying also a Quality-Assurance filtering. It internally relies
    on the SelectGHOSTStationsMonth function that select GHOST stations over
    a specific month.

    Parameters
    ----------
    htime : date array
        Entire hourly date array of interest. Only useful if the user is not interested
        entire months.        

    pollutant : str
        Pollutant requested. Should follow the GHOST nomenclatura.

    path_ghost : str
        Path of the GHOST root directory

    drop_qa : list of int/float, default=None 
        List of Quality-Assurance (QA) flags to filter. All measurements with one of 
        these flags will be removed

    extent : list of float, default=None 
        Geographical extent of the domain where stations should be selected :
        lon_min, lon_max, lat_min, lat_max

    station_classification : list of str, default=None 
        List of station classification to select

    area_classification : list of str, default=None
        List of area classification to select
    
    n_jobs : int, default=1
        Number of jobs to run in parallel. If n_jobs > 1, then the function will
        run in parallel, each month being run in parallel on a specific job.

    verbose : bool, default=False
        To print intermediate information

    Output
    ------
    output : Pandas dataframe with information on the selected stations:
        - station_reference  
        - longitude  
        - latitude  
        - altitude   
        - station_classification 
        - area_classification
        - country

    Example
    -------
    >>> from pyeslib import PrepareTimeArrays,SelectGHOSTStationsMonth
    >>> htime = PrepareTimeArrays('2019012500','2019123123', ['h'])['htime']
    >>> network, ghost_version = 'EEA_AQ_eReporting', '1.3.3'
    >>> res = SelectGHOSTStations(htime,
    ...                           pollutant='sconco3',
    ...                           path_ghost='/gpfs/projects/bsc32/AC_cache/obs/ghost/{}/{}'.format(network,ghost_version),
    ...                           extent=[-10,10, 45,50],
    ...                           drop_qa=[0, 1, 2, 3, 6, 8, 10, 12, 13, 14, 17, 18, 22, 25, 40, 41, 42],
    ...                           station_classification=['background', 'point_source-industrial'],
    ...                           area_classification=['rural'],
    ...                           n_jobs=4)
    >>> print(res)
      station_reference  longitude   latitude  altitude   station_classification area_classification
      0   STA-CH0002R_CL(IMC)   6.944480  46.813100     489.0               background               rural
      1   STA-CH0003R_CL(IMC)   8.904680  47.479800     538.0               background               rural
      2   STA-CH0004R_CL(IMC)   6.979150  47.049500    1136.0               background               rural
      3   STA-CH0005R_CL(IMC)   8.463330  47.067400    1031.0               background               rural
      4   STA-CH0019A_CL(IMC)   9.394380  47.406700     915.0               background               rural
      5   STA-CH0024A_CL(IMC)   7.148330  46.139000     460.0               background               rural
      6   STA-CH0033A_CL(IMC)   8.933940  46.160300     203.0               background               rural
      7   STA-CH0051A_CL(IMC)   6.005180  46.162600     427.0               background               rural
      8   STA-CH0053R_CL(IMC)   8.175440  47.189600     797.0               background               rural
      9   STA.DE_DERP017_CAPS   7.826407  49.270263     600.0               background               rural
      10  STA.IT1464A_CL(IMC)   9.556389  45.497500     115.0               background               rural
      11  STA.IT1736A_CL(IMC)   8.915000  45.040833      74.0               background               rural
      12  STA.IT1961A_CL(IMC)   8.255900  46.314800    1639.0               background               rural
      13  STA.IT1963A_CL(IMC)   7.245400  45.430200    1576.0               background               rural
      14  STA.IT1964A_CL(IMC)   9.666940  45.235000      63.0               background               rural
      15  STA.IT2014A_CL(IMC)   9.286111  46.015833     211.0               background               rural
      16  STA.IT2063A_CL(IMC)   9.940278  45.148333      48.0  point_source-industrial               rural

    """
    # Load librairies
    import time
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    import copy

    # Check type of inputs
    assert GetType(htime) in ['pandas.core.indexes.datetimes.DatetimeIndex',
                              'pd.core.indexes.datetimes.DatetimeIndex'],'htime should be a pandas datetime'
    assert GetType(pollutant)=='str','pollutant should be a string'
    assert GetType(path_ghost)=='str','path_ghost should be a string'
    if GetType(drop_qa)=='list':
        assert GetType(drop_qa[0])=='int','drop_qa components should be integers'
    assert GetType(extent) in ['NoneType','list'],'extent should be a list'
    if GetType(extent)=='list': 
        assert GetType(extent[0]) in ['int','float','numpy.float32','np.float32','numpy.float64','np.float64'],'extent components should be int or float'
        assert len(extent)==4,'extent should have 4 components'
    assert GetType(station_classification) in ['NoneType','list'],'station_classification should be None or list'
    if GetType(station_classification)=='list':
        assert GetType(station_classification[0])=='str','station_classification components should be string'
    assert GetType(area_classification) in ['NoneType','list'],'area_classification should be None or list'
    if GetType(area_classification)=='list':
        assert GetType(area_classification[0])=='str','area_classification components should be string'   
    assert GetType(n_jobs)=='int','n_jobs should be an integer'
    assert n_jobs > 0,'n_jobs should be strictly positive' 
    assert GetType(verbose)=='bool','verbose should be a boolean'
    
    # Get list of month in YYYYMM format
    list_yyyymm = np.unique(htime.strftime('%Y%m'))

    # Select stations
    if n_jobs>1:
        #(parallel)
        import multiprocessing
        jobargs = [(yyyymm,
                    htime,
                    pollutant,
                    path_ghost,
                    drop_qa,
                    extent,
                    station_classification,
                    area_classification,
                    timescale,
                    verbose) for iyyyymm, yyyymm in enumerate(list_yyyymm)]
        pool = multiprocessing.Pool(n_jobs)
        print('N(jobs) = {}'.format(len(jobargs)))
        allres = pool.starmap(SelectGHOSTStationsMonth, jobargs)
        pool.close()
        pool.join()
        for iyyyymm,res in enumerate(allres):
            output = copy.deepcopy(res) if iyyyymm==0 else pd.concat([output,res]).drop_duplicates(ignore_index=True).reset_index(drop=True)   
    else:
        #(sequetial)
        for iyyyymm, yyyymm in enumerate(list_yyyymm):
            start = time.time() 
            res = SelectGHOSTStationsMonth(yyyymm,
                                           htime,
                                           pollutant=pollutant,
                                           path_ghost=path_ghost,
                                           drop_qa=drop_qa,
                                           extent=extent,
                                           station_classification=station_classification,
                                           area_classification=area_classification,
                                           timescale=timescale,
                                           verbose=verbose)
            if verbose: print('                  (fun:{},{})'.format(str(timedelta(seconds=time.time()-start)),yyyymm))  
            output = copy.deepcopy(res) if iyyyymm==0 else pd.concat([output,res]).drop_duplicates(ignore_index=True).reset_index(drop=True)
            if verbose: print('                  (post::{},{})'.format(str(timedelta(seconds=time.time()-start)),yyyymm)) 
    return(output)


def SelectGHOSTStationsMonth(yyyymm,
                             htime=None,
                             pollutant=None,
                             path_ghost=None,
                             drop_qa=None,
                             extent=None, 
                             station_classification=None, 
                             area_classification=None,
                             timescale=['hourly'], 
                             verbose=False):
    """ Sub-function of SelectGHOSTStations for selecting GHOST stations,
    focusing on a specific month.

    See documentation of SelectGHOSTStations for more detail information.
    """
    if verbose: print(yyyymm)

    # Load libraries
    import time ; start = time.time()
    import numpy as np
    import pandas as pd
    import xarray as xr
    import warnings
    import os
    from datetime import timedelta
    
    # Check type of inputs
    assert GetType(yyyymm)=='str','yyyymm should be a string'
    assert GetType(htime) in ['pandas.core.indexes.datetimes.DatetimeIndex',
                              'pd.core.indexes.datetimes.DatetimeIndex'],'htime should be a pandas datetime'
    assert GetType(pollutant)=='str','pollutant should be a string'
    assert GetType(path_ghost)=='str','path_ghost should be a string'
    assert GetType(drop_qa) in ['NoneType','list'],'drop_qa should be a list'
    if GetType(drop_qa)=='list':
        assert GetType(drop_qa[0])=='int','drop_qa components should be integers'
    assert GetType(extent) in ['NoneType','list'],'extent should be a list'
    if GetType(extent)=='list': 
        assert GetType(extent[0]) in ['int','float','numpy.float32','np.float32','numpy.float64','np.float64'],'extent components should be int or float'
        assert len(extent)==4,'extent should have 4 components'
    assert GetType(station_classification) in ['NoneType','list'],'station_classification should be None or list'
    if GetType(station_classification)=='list':
        assert GetType(station_classification[0])=='str','station_classification components should be string'
    assert GetType(area_classification) in ['NoneType','list'],'area_classification should be None or list'
    if GetType(area_classification)=='list':
        assert GetType(area_classification[0])=='str','area_classification components should be string'   
    assert GetType(verbose)=='bool','verbose should be a boolean'
        
    # Define the information of interest for the output
    selected_station_reference = []
    selected_longitude = []
    selected_latitude = []
    selected_altitude = []
    selected_station_classification = []
    selected_area_classification = []
    selected_measurement_methodology = []
    selected_country = []

    for its,ts in enumerate(timescale):
        
        # Define the file of the GHOST monthly netcdf files
        ncfile = '{}/{}/{}/{}_{}.nc'.format(path_ghost,
                                            ts,
                                            pollutant,
                                            pollutant,
                                            yyyymm)

        if os.path.exists(ncfile)==True:
            # Read dataset
            startread = time.time()
            if verbose: print(ncfile)
            ds = xr.open_dataset(ncfile)
            ds.close()
            if verbose: print('                  (read:{})'.format(str(timedelta(seconds=time.time()-startread))))
            #ds.load()
            #if verbose: print('                  (load:{})'.format(str(timedelta(seconds=time.time()-startread))))               
            
            # Identify requested filters
            filters = ''
            if not extent is None: filters += 'extent|'
            if not station_classification is None: filters += 'station_classification|'
            if not area_classification is None: filters += 'area_classification|'
            
            # Loop on selection filters
            if verbose : print('| Filters list: {}'.format(filters.split('|')[:-1]))
            for ifilt,filt in enumerate(filters.split('|')[:-1]):
                if filt=='extent':
                    # Find stations over target region
                    wfilter = np.where((ds.longitude.values >= extent[0]) &
                                       (ds.longitude.values <= extent[1]) &
                                       (ds.latitude.values  >= extent[2]) &
                                       (ds.latitude.values  <= extent[3]))[0]
                elif filt=='station_classification':
                    # Find stations with requested station classifications
                    wfilter = []
                    for xclass in station_classification:
                        wfilter += list(np.where(ds.station_classification.values==xclass)[0])
                elif filt=='area_classification':
                    # Find stations with requested area classifications
                    wfilter = []
                    for xclass in area_classification:
                        wfilter += list(np.where(ds.area_classification.values==xclass)[0])

                # Filter
                if verbose: print('{:25s}({}) : keeping {} stations'.format(filt,yyyymm,len(wfilter)))
                if len(wfilter)!=0:
                    ds = ds[dict(station=wfilter)]

            # Quality-Assurance (QA) filtering
            if verbose : print('| Start QA filtering')
            if not drop_qa is None:
                startqa = time.time()
                conc = ds[pollutant].values
                conc[np.isin(ds.qa.values,drop_qa).any(axis=2)] = np.nan
                ds[pollutant][dict(station=range(ds.dims['station']),time=range(ds.dims['time']))] = conc
                print('                  (qa-duration:{})'.format(str(timedelta(seconds=time.time()-startqa))))
            
            # Keep only stations with at least one data during the period of interest
            wtime = np.intersect1d(ds.time.values,htime, return_indices=True)[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                wstat = np.where(np.isfinite(ds[pollutant][dict(time=wtime)].mean('time').values))[0]
            
            if len(wstat)!=0:
                ds = ds[dict(station=wstat)]
                if verbose: print('{:25s}({}) : keeping {} stations (qa:{})'.format('QA-filtering',yyyymm,len(wstat),str(drop_qa)))

                # Get list of stations
                selected_station_reference += [str(i) for i in ds.station_reference.values]
                selected_longitude += list(ds.longitude.values)
                selected_latitude += list(ds.latitude.values)
                selected_altitude += list(ds.altitude.values)
                selected_station_classification += [str(i) for i in ds.station_classification.values]
                selected_area_classification += [str(i) for i in ds.area_classification.values]
                selected_country += [str(i) for i in ds.country.values]
                selected_measurement_methodology += [str(i) for i in ds.measurement_methodology.values]

    # remove duplicates
    selected_station_reference_unique = np.unique(np.array(selected_station_reference))
    w = [np.where(np.array(selected_station_reference)==x)[0][0] for x in selected_station_reference_unique]
    if len(w)!=0:
        selected_station_reference = np.array(selected_station_reference)[w]
        selected_longitude = np.array(selected_longitude)[w]
        selected_latitude = np.array(selected_latitude)[w]
        selected_altitude = np.array(selected_altitude)[w]
        selected_station_classification = np.array(selected_station_classification)[w]
        selected_area_classification = np.array(selected_area_classification)[w]
        selected_country = np.array(selected_country)[w]
        selected_measurement_methodology = np.array(selected_measurement_methodology)[w]
        
    # Build output dataframe
    output = pd.DataFrame({'station':selected_station_reference,
                           'station_longitude':selected_longitude,
                           'station_latitude':selected_latitude,
                           'station_altitude':selected_altitude,
                           'station_classification':selected_station_classification,
                           'station_area_classification':selected_area_classification,
                           'station_country':selected_country,
                           'measurement_methodology':selected_measurement_methodology})


    print('{} (duration:{})'.format(yyyymm,str(timedelta(seconds=time.time()-start))))
    return(output)
