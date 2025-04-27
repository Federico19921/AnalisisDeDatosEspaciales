import numpy as np
import re
import os
import matplotlib.pyplot as plt

import geopandas as gpd
from zipfile import ZipFile
#from shapely.geometry.polygon import Polygon
from shapely.geometry import shape

import rasterio
from rasterio.mask import mask
from rasterio.plot import show


def nequalize(array,p=5,nodata=None):
    """
    normalize and equalize a single band image
    """
    if len(array.shape)==2:
        vmin=np.percentile(array[array!=nodata],p)
        vmax=np.percentile(array[array!=nodata],100-p)
        eq_array = (array-vmin)/(vmax-vmin)
        eq_array[eq_array>1]=1
        eq_array[eq_array<0]=0
    elif len(array.shape)==3:
        eq_array = np.empty_like(array, dtype=float)
        for i in range(array.shape[0]):
            eq_array[i]=nequalize(array[i], p=p, nodata=nodata)
    return eq_array

def plot_rgb(array, band_list , p = 0, nodata = None, figsize = (12,6), title = None):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, 
    una lista de índices correspondientes a las bandas que queremos usar, 
    en el orden que deben estar (ej: [1,2,3]), y un parámetro p que es opcional 
    que es el percentil de equalización.
    
    Por defecto tambien asigna un tamaño de figura en (12,6), que también puede ser modificado.
    
    Devuelve solamente un ploteo, no modifica el arreglo original.
    Nota: array debe ser una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    if not title:
        title = f'Combinación {band_list} \n (percentil {p}%)'
        
    img = nequalize(array[band_list], p=p, nodata=nodata)
    plt.figure(figsize = figsize)
    plt.title(title , size = 20)
    show(img)
    plt.show()    
    
def guardar_GTiff(fn, crs, transform, mat, meta=None, nodata=None, bandnames=[]):
    if len(mat.shape)==2:
        count=1
    else:
        count=mat.shape[0]

    if not meta:
        meta = {}

    meta['driver'] = 'GTiff'
    meta['height'] = mat.shape[-2]
    meta['width'] = mat.shape[-1]
    meta['count'] = count
    meta['crs'] = crs
    meta['transform'] = transform

    if 'dtype' not in meta: #if no datatype is specified, use float32
        meta['dtype'] = np.float32
    

    if nodata==None:
        pass
    else:
        meta['nodata'] = nodata

    with rasterio.open(fn, 'w', **meta) as dst:
        if count==1: #es una matriz bidimensional, la guardo
            dst.write(mat.astype(meta['dtype']), 1)
            if bandnames:
                dst.set_band_description(1, bandnames[0])
        else: #es una matriz tridimensional, guardo cada banda
            for b in range(count):
                dst.write(mat[b].astype(meta['dtype']), b+1)
            for b,bandname in enumerate(bandnames):
                dst.set_band_description(b+1, bandname)#   
                

def compute_mbb(fn, snap_to_grid = True, grid_step = 10):
    """dado archivo vectorial (shp, geojson, etc) 
    calcula el mínimo rectángulo que contenga 
    su primer objeto, usando vértices 
    en una grilla de paso dado."""

    gdf = gpd.read_file(fn)
    first_geom = gdf.iloc[0]['geometry'] #miro solo la primer geometría del archivo

    mX, mY, MX, MY = first_geom.bounds
    if snap_to_grid:
        mX = grid_step*(np.floor(mX/grid_step))
        MX = grid_step*(np.ceil(MX/grid_step))
        mY = grid_step*(np.floor(mY/grid_step))
        MY = grid_step*(np.ceil(MY/grid_step))

    mbb = shape({'type': 'Polygon',
          'coordinates': [((mX, MY),
                           (MX, MY),
                           (MX, mY),
                           (mX, mY),
                           (mX, MY))]})
    return mbb
    
# Procesar Sentinel2
def extract_10m_bands_Sentinel2(img_data_dir, mbb=None, compute_ndvi = True, verbose = True):
    """dado un directorio con las bandas de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes"""
    
    ls = os.listdir(img_data_dir)
    band_names = ['B02.','B03.', 'B04.', 'B08.'] 
    bands = []
    for b in band_names:
        try:
            fn = [fn for fn in ls if b in fn][0]
        except:
            print(f"Banda {b} no encontrada en {img_data_dir}.")
        if verbose: print(f"Leyendo {fn}.")
        
        fn = os.path.join(img_data_dir,fn)
        with rasterio.open(fn) as src:
            crs=src.crs #recuerdo el sistema de referencia para poder grabar
            if mbb: #si hay mbb hago un clip
                array, out_transform = mask(src, [mbb])
            else: #si no, uso la imagen entera
                array = src.read()
                out_transform = src.transform
        bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
    if compute_ndvi:
        if verbose: print("Computando NDVI.")
        bands.append((bands[3]-bands[2])/(bands[3]+bands[2]))
    return np.stack(bands), crs, out_transform

# Procesar Sentinel2
def extract_10m_bands_Sentinel2_ZIP(zipfilename, mbb=None, compute_ndvi = True, verbose = True):
    """dado un zip de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes"""
    
    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 10m resolution bands: 02, 03, 04 and 08
    tileREXP = re.compile(r'.*_B(02|03|04|08).jp2$')
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if re.match(tileREXP,x)]
        bandfns.sort()
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                crs=src.crs #recuerdo el sistema de referencia para poder grabar
                if mbb: #si hay mbb hago un clip
                    array, out_transform = mask(src, [mbb], crop=True)
                else: #si no, uso la imagen entera
                    array = src.read()
                    out_transform = src.transform
            bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
    if compute_ndvi:
        if verbose: print('Computando NDVI.')
        bands.append((bands[3]-bands[2])/(bands[3]+bands[2]))
    return np.stack(bands), crs, out_transform
    
