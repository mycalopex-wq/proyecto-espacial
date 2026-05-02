# -*- coding: utf-8 -*-
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import geopandas as gpd
import numpy as np
import pandas as pd
import tempfile, zipfile, os, re, io, gc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import random
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx 

# Configuración de página optimizada
st.set_page_config(page_title="Geo-Analizador Satelital Lite", layout="wide")
st.title("Plataforma de Análisis Satelital (Cloud Edition)")

DOG_GIF_URL = "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmZlZHV1djJ4NnVuNWRod2JweGIwY3ZoamZkdnV2bGQ3ZXpxcG84MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/f9vsEmv4NA9ry/giphy.gif"
MAX_FILE_SIZE_MB = 100

# -----------------------------
# 1. FUNCIONES DE UTILIDAD
# -----------------------------
def check_file_size(file):
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"El archivo '{file.name}' supera el límite de {MAX_FILE_SIZE_MB}MB. Por favor, redúcelo o recórtalo antes de subirlo.")
        return False
    return True

@st.cache_data
def process_vector_file(uploaded_file):
    if uploaded_file.name.endswith('.zip'):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            z.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.endswith(".shp"): return os.path.join(root, f)
    elif uploaded_file.name.endswith('.gpkg'):
        temp_gpkg = tempfile.NamedTemporaryFile(delete=False, suffix=".gpkg")
        temp_gpkg.write(uploaded_file.getvalue()); temp_gpkg.close()
        return temp_gpkg.name
    return None

@st.cache_data
def load_raster_lite(uploaded_file, target_crs_str=None):
    t_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    t_raw.write(uploaded_file.getvalue()); t_raw.close()
    
    if target_crs_str:
        target_crs = rasterio.crs.CRS.from_string(target_crs_str)
        with rasterio.open(t_raw.name) as src:
            if src.crs != target_crs:
                transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height, "compress": 'lzw'})
                reproj_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
                with rasterio.open(reproj_path, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                                  src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=target_crs,
                                  resampling=Resampling.nearest)
                return reproj_path
    return t_raw.name

def parse_scene_name(filename):
    match = re.search(r'^(\d{4}-\d{2}-\d{2})', filename)
    return f"Escena {match.group(1)}" if match else filename[:15]

# -----------------------------
# 2. MOTOR DE RENDERIZADO LITE
# -----------------------------
def generar_mapa_tematico(raster_path, modo, bandas, escala, title):
    with rasterio.open(raster_path) as src:
        data = src.read().astype(float)
        data[data <= 0] = np.nan
        ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if modo == "NDVI":
            nir, red = data[bandas['n']-1], data[bandas['r']-1]
            ndvi = (nir - red) / (nir + red + 1e-6)
            im = ax.imshow(ndvi, cmap='RdYlGn', extent=ext, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label="NDVI")
        elif modo == "RGB":
            r, g, b = data[bandas['r']-1]/escala, data[bandas['g']-1]/escala, data[bandas['b']-1]/escala
            rgb = np.dstack([np.nan_to_num(np.clip(r,0,1)), np.nan_to_num(np.clip(g,0,1)), np.nan_to_num(np.clip(b,0,1))])
            ax.imshow(rgb, extent=ext)
            
        ax.set_title(title, weight='bold')
        try: cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=src.crs.to_string(), alpha=0.5)
        except: pass
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig); del data; gc.collect()
        return buf.getvalue()

# -----------------------------
# 3. SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Configuración Lite")
    st.info("Ideal para análisis de cuencas como el Biobío o Petorca usando Sentinel-2.")
    
    sat_files = st.file_uploader("Subir Imágenes Satelitales (.tif)", type=["tif"], accept_multiple_files=True)
    v_file = st.file_uploader("Subir Vector Opcional (ZIP/GPKG)", type=["zip", "gpkg"])
    
    if v_file:
        st.session_state.raw_gdf = gpd.read_file(process_vector_file(v_file))
        st.session_state.col_clase = st.selectbox("Columna Clase:", [c for c in st.session_state.raw_gdf.columns if c != 'geometry'])

    st.divider()
    with st.expander("Configuración de Bandas"):
        scale = st.number_input("Factor Escala (ej. 10000)", 1.0, 20000.0, 10000.0)
        b_r = st.number_input("Banda Roja", 1, 12, 3)
        b_g = st.number_input("Banda Verde", 1, 12, 2)
        b_b = st.number_input("Banda Azul", 1, 12, 1)
        b_n = st.number_input("Banda NIR", 1, 12, 4)
        bandas_dict = {'r': b_r, 'g': b_g, 'b': b_b, 'n': b_n}

    if st.button("Ejecutar Análisis", use_container_width=True):
        if sat_files:
            valid_files = [f for f in sat_files if check_file_size(f)]
            if valid_files:
                st.session_state.data_lite = {}
                for f in valid_files:
                    name = parse_scene_name(f.name)
                    path = load_raster_lite(f)
                    st.session_state.data_lite[name] = {'path': path, 'name': name}
                st.session_state.run_lite = True

# -----------------------------
# 4. DASHBOARD
# -----------------------------
if st.session_state.get("run_lite"):
    names = list(st.session_state.data_lite.keys())
    tabs = st.tabs([f"{n}" for n in names] + ["Comparación Global"])

    for idx, name in enumerate(names):
        with tabs[idx]:
            d = st.session_state.data_lite[name]
            
            if 'maps' not in d:
                with st.spinner(f"Analizando {name}..."):
                    st.image(DOG_GIF_URL, width=150)
                    d['maps'] = {
                        'RGB': generar_mapa_tematico(d['path'], "RGB", bandas_dict, scale, f"Color Real - {name}"),
                        'NDVI': generar_mapa_tematico(d['path'], "NDVI", bandas_dict, scale, f"NDVI - {name}")
                    }
            
            c1, c2 = st.columns(2)
            c1.image(d['maps']['RGB'], use_container_width=True)
            c2.image(d['maps']['NDVI'], use_container_width=True)

            if 'col_clase' in st.session_state:
                st.divider()
                st.subheader("Estadísticas por Zona")
                st.info("Vector detectado: Listo para extraer firmas espectrales en la versión completa.")

    with tabs[-1]:
        st.header("Resumen del Proyecto")
        st.write(f"Total de escenas analizadas: {len(names)}")
