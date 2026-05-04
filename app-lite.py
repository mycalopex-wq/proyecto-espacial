# -*- coding: utf-8 -*-
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import geopandas as gpd
import numpy as np
import pandas as pd
import tempfile, zipfile, os, re, io, gc
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import folium
from streamlit_folium import st_folium
import plotly.express as px
import random
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx 

st.set_page_config(page_title="Plataforma de Analisis Lite", layout="wide")
st.title("Plataforma de Análisis Satelital (Lite Edition)")
st.info("Versión optimizada para la nube. Límite de 100MB por archivo raster. Ideal para análisis de cuencas y ecosistemas con Sentinel-2 o Landsat.")

# -----------------------------
# 1. FUNCIONES DE PROCESAMIENTO
# -----------------------------
MAX_FILE_SIZE_MB = 100

def check_size(f):
    if f and f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.sidebar.error(f"El archivo {f.name} pesa más de {MAX_FILE_SIZE_MB}MB. Por favor, recórtalo antes de subirlo.")
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
def load_vector_preview(vector_file):
    path = process_vector_file(vector_file)
    return gpd.read_file(path)

def parse_scene_name(filename):
    match = re.search(r'^(\d{4}-\d{2}-\d{2})_([^_]+)', filename)
    if match: return f"{match.group(2).replace('-', ' ').title()} ({match.group(1)})"
    return os.path.splitext(filename)[0][:15]

def inicializar_base_lite(sat_file, master_crs, master_gdf, col_clase):
    data = {}
    t_sat = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    t_sat.write(sat_file.getvalue()); t_sat.close()
    
    data['sat_path'] = t_sat.name
    
    if master_gdf is not None and col_clase:
        with rasterio.open(data['sat_path']) as src:
            gdf_cortado = gpd.clip(master_gdf, box(*src.bounds))
            if not gdf_cortado.empty:
                if gdf_cortado.crs.is_geographic: gdf_area = gdf_cortado.to_crs(epsg=3857)
                else: gdf_area = gdf_cortado.copy()
                gdf_cortado['area_m2'] = gdf_area.geometry.area
                data['gdf'] = gdf_cortado
                data['gdf_diss'] = gdf_cortado.dissolve(by=col_clase, aggfunc={'area_m2': 'sum'}).reset_index()
            else:
                data['gdf'] = None; data['gdf_diss'] = None
    else:
        data['gdf'] = None; data['gdf_diss'] = None
        
    return data

def calcular_firmas_lite(data_dict, col_clase, sat_scale, s_idx_list, sat_name):
    if data_dict['gdf_diss'] is None: return pd.DataFrame()
    resultados = []
    s_b, s_g, s_r, s_re, s_n, s_s1, s_s2 = s_idx_list
    
    with rasterio.open(data_dict['sat_path']) as sat_src:
        for _, row in data_dict['gdf_diss'].iterrows():
            pts, bbox, intentos = [], row.geometry.bounds, 0
            while len(pts) < 100 and intentos < 1500:
                p = Point(random.uniform(bbox[0], bbox[2]), random.uniform(bbox[1], bbox[3]))
                if p.within(row.geometry): pts.append(p)
                intentos += 1
            if not pts: continue
            
            coords = [(pt.x, pt.y) for pt in pts]
            m_sat = np.array(list(sat_src.sample(coords))).astype(float) / sat_scale
            m_sat[m_sat <= 0] = np.nan
            m_sat = m_sat[~np.isnan(m_sat).any(axis=1)]
            
            if len(m_sat) > 0:
                f_sat = np.nanmean(m_sat, axis=0)
                b_map = {s_b:"Azul", s_g:"Verde", s_r:"Rojo", s_re:"Red Edge", s_n:"NIR", s_s1:"SWIR 1", s_s2:"SWIR 2"}
                for b in range(sat_src.count):
                    if (b+1) in b_map and (b+1) > 0: 
                        resultados.append({'Cobertura': row[col_clase], 'Banda': b_map[b+1], 'Sensor': sat_name, 'Reflectancia': f_sat[b]})
    return pd.DataFrame(resultados)

def generar_mapa_crudo_lite(d, modo, s_list, scale, esc_name):
    s_b, s_g, s_r, s_re, s_n, s_s1, s_s2 = s_list
    with rasterio.open(d['sat_path']) as base:
        ext = [base.bounds.left, base.bounds.right, base.bounds.bottom, base.bounds.top]
        def get_b(idx):
            out = np.full((base.height, base.width), np.nan, dtype=np.float32)
            if idx > 0 and idx <= base.count: 
                out = base.read(int(idx)).astype(float) / scale
            out[out <= 0] = np.nan; return out
        def norm(a):
            if np.isnan(a).all(): return a
            p2, p98 = np.nanpercentile(a, [2, 98])
            return (np.clip(a, p2, p98) - p2) / (p98 - p2 + 1e-6)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if modo == "NDVI":
            nir, red = get_b(s_n), get_b(s_r)
            ndvi = (nir - red) / (nir + red + 1e-6)
            
            if not np.isnan(ndvi).all():
                p2, p98 = np.nanpercentile(ndvi, [2, 98])
                vmin = min(0.0, p2) if p2 < 0 else p2
                im = ax.imshow(ndvi, cmap='RdYlGn', extent=ext, vmin=vmin, vmax=p98)
            else:
                im = ax.imshow(ndvi, cmap='RdYlGn', extent=ext, vmin=-1, vmax=1)
                
            plt.colorbar(im, ax=ax, label="NDVI")
            del nir, red, ndvi
            
        elif "RGB" in modo:
            r, g, b = norm(get_b(s_r)), norm(get_b(s_g)), norm(get_b(s_b))
            ax.imshow(np.dstack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)]), extent=ext)
            del r, g, b
        elif "Falso Color SWIR" in modo:
            r, g, b = norm(get_b(s_s1)), norm(get_b(s_n)), norm(get_b(s_r))
            ax.imshow(np.dstack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)]), extent=ext)
            del r, g, b
            
        ax.set_title(f"{modo} - {esc_name}", pad=20, fontsize=12, weight='bold')
        try: cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=base.crs.to_string(), alpha=0.5)
        except: pass
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig); gc.collect()
        return buf.getvalue()

# -----------------------------
# 3. SIDEBAR (LITE)
# -----------------------------
with st.sidebar:
    st.header("1. Archivo Vectorial (Opcional)")
    vector_file = st.file_uploader("Capa de Zonas (ZIP o GPKG)", type=["zip", "gpkg"])
    if vector_file:
        preview_gdf = load_vector_preview(vector_file)
        st.session_state.raw_gdf = preview_gdf
        
        columna_elegida = st.selectbox("Columna Clase:", [c for c in preview_gdf.columns if c != 'geometry'])
        st.session_state.col_clase = columna_elegida
    else:
        st.session_state.raw_gdf = None
        st.session_state.col_clase = None

    st.divider()
    st.header("2. Gestión de Escenas")
    num_escenas = st.number_input("Cantidad de Escenas", 1, 10, 1)
    archivos_escenas = []
    for i in range(1, num_escenas + 1):
        with st.expander(f"Escena {i}", expanded=(i==1)):
            archivos_escenas.append({
                "id": i, 
                "sat": st.file_uploader(f"Imagen Satelital (E{i})", type=["tif"])
            })

    st.divider()
    with st.expander("3. Configuración del Satélite", expanded=True):
        sat_name = st.text_input("Sensor", "Sentinel-2")
        sat_scale = st.number_input("Factor Escala", 10000.0)
        c1, c2 = st.columns(2)
        with c1:
            s_b = st.number_input("Banda Azul", 1)
            s_g = st.number_input("Banda Verde", 2)
            s_r = st.number_input("Banda Roja", 3)
            s_re = st.number_input("Red Edge", 4)
        with c2:
            s_n = st.number_input("Banda NIR", 5)
            s_swir1 = st.number_input("SWIR 1", 0)
            s_swir2 = st.number_input("SWIR 2", 0)
        s_idx_list = [s_b, s_g, s_r, s_re, s_n, s_swir1, s_swir2]

    st.divider()
    if st.button("Ejecutar Análisis Lite", use_container_width=True):
        first_valid_raster = None
        for e in archivos_escenas:
            if e['sat'] and check_size(e['sat']): first_valid_raster = e['sat']; break
            
        if first_valid_raster:
            with MemoryFile(first_valid_raster.getvalue()) as mem: master_crs = mem.open().crs
            master_gdf = st.session_state.raw_gdf.to_crs(master_crs) if st.session_state.raw_gdf is not None else None
            
            st.session_state.data_escenas = {}
            for e in archivos_escenas:
                if e['sat'] and check_size(e['sat']):
                    name = parse_scene_name(e['sat'].name)
                    st.session_state.data_escenas[name] = inicializar_base_lite(e['sat'], master_crs, master_gdf, st.session_state.col_clase)
            st.session_state.analisis_listo = True
        else:
            st.error("Sube al menos un archivo satelital válido (menor a 100MB).")

    if st.button("Reiniciar Entorno"): st.session_state.clear(); st.rerun()

# -----------------------------
# 4. RENDERIZADO PRINCIPAL
# -----------------------------
if st.session_state.get("analisis_listo"):
    names = list(st.session_state.data_escenas.keys())
    tabs = st.tabs([f"{n}" for n in names])
    
    for idx, name in enumerate(names):
        with tabs[idx]:
            d = st.session_state.data_escenas[name]
            if 'pre_m' not in d:
                with st.spinner(f"Analizando imagen {name}..."):
                    df_f = calcular_firmas_lite(d, st.session_state.col_clase, sat_scale, s_idx_list, sat_name)
                    d['df_f'] = df_f
                    
                    d['pf'] = {}
                    if not df_f.empty:
                        # Agrupación para la firma global
                        df_global = df_f.groupby(['Cobertura', 'Banda'])['Reflectancia'].mean().reset_index()
                        
                        # Generación del gráfico global
                        fig_global = px.line(df_global, x="Banda", y="Reflectancia", color="Cobertura", markers=True, title="Comparación Global de Coberturas")
                        fig_global.update_xaxes(categoryorder='array', categoryarray=["Azul", "Verde", "Rojo", "Red Edge", "NIR", "SWIR 1", "SWIR 2"])
                        d['pf_global'] = fig_global

                        # Generación de gráficos individuales
                        for cob in df_global['Cobertura'].unique():
                            sub_f = df_global[df_global['Cobertura'] == cob]
                            fig = px.line(sub_f, x="Banda", y="Reflectancia", markers=True, title=cob)
                            fig.update_xaxes(categoryorder='array', categoryarray=["Azul", "Verde", "Rojo", "Red Edge", "NIR", "SWIR 1", "SWIR 2"])
                            d['pf'][cob] = fig
                            
                    d['pre_m'] = {
                        'RGB': generar_mapa_crudo_lite(d, "RGB", s_idx_list, sat_scale, name),
                        'NDVI': generar_mapa_crudo_lite(d, "NDVI", s_idx_list, sat_scale, name)
                    }
                    if s_swir1 > 0:
                        d['pre_m']['Falso Color SWIR'] = generar_mapa_crudo_lite(d, "Falso Color SWIR", s_idx_list, sat_scale, name)
            
            if d['gdf'] is not None:
                col_m, col_t = st.columns([2, 1])
                with col_m:
                    gdf_map = d['gdf'].to_crs(epsg=4326)
                    m = folium.Map(location=[gdf_map.total_bounds[[1,3]].mean(), gdf_map.total_bounds[[0,2]].mean()], zoom_start=14)
                    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)
                    folium.GeoJson(gdf_map).add_to(m); st_folium(m, width=700, height=400, key=f"f_{name}")
                with col_t:
                    st.plotly_chart(px.pie(d['gdf_diss'], values='area_m2', names=st.session_state.col_clase, title="Distribución de Zonas (Hectáreas)"), use_container_width=True)
            else:
                st.info("Visualizando imagen completa. Si deseas estadísticas por zona o firmas espectrales, sube un archivo vectorial.")
            
            st.divider()
            
            t_sub = st.tabs(["Cartografía Temática", "Firmas Espectrales"])
            with t_sub[0]:
                if s_swir1 > 0:
                    cols = st.columns(3)
                    cols[0].image(d['pre_m']['RGB'], use_container_width=True, caption="Composición Color Real")
                    cols[1].image(d['pre_m']['NDVI'], use_container_width=True, caption="Índice de Vegetación (NDVI)")
                    cols[2].image(d['pre_m']['Falso Color SWIR'], use_container_width=True, caption="Falso Color (SWIR-NIR-Rojo)")
                else:
                    cols = st.columns(2)
                    cols[0].image(d['pre_m']['RGB'], use_container_width=True, caption="Composición Color Real")
                    cols[1].image(d['pre_m']['NDVI'], use_container_width=True, caption="Índice de Vegetación (NDVI)")

            with t_sub[1]:
                if not d['df_f'].empty:
                    # Mostrar gráfico global primero
                    st.plotly_chart(d['pf_global'], use_container_width=True)
                    st.divider()
                    
                    st.markdown("### Firmas Individuales por Cobertura")
                    cols = st.columns(3)
                    for i, c in enumerate(d['pf'].keys()):
                        cols[i%3].plotly_chart(d['pf'][c], use_container_width=True)
                else: 
                    st.warning("Requiere subir un archivo vectorial y seleccionar una columna de clase para extraer firmas espectrales.")
