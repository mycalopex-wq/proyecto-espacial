# -*- coding: utf-8 -*-
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.mask import mask
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

st.set_page_config(page_title="Plataforma de Analisis Espacial", layout="wide")
st.title("Plataforma de Análisis Espacial y Multitemporal")

DOG_GIF_URL = "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmZlZHV1djJ4NnVuNWRod2JweGIwY3ZoamZkdnV2bGQ3ZXpxcG84MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/f9vsEmv4NA9ry/giphy.gif"

# -----------------------------
# 1. FUNCIONES DE PROCESAMIENTO
# -----------------------------
def check_size(f):
    if f and f.size > 100 * 1024 * 1024:
        st.sidebar.error(f"El archivo {f.name} pesa más de 100MB. Por favor, recórtalo antes de subirlo.")
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
                if f.endswith(".shp"):
                    return os.path.join(root, f)
    elif uploaded_file.name.endswith('.gpkg'):
        temp_gpkg = tempfile.NamedTemporaryFile(delete=False, suffix=".gpkg")
        temp_gpkg.write(uploaded_file.getvalue())
        temp_gpkg.close()
        return temp_gpkg.name
    return None

@st.cache_data
def load_vector_preview(vector_file):
    path = process_vector_file(vector_file)
    return gpd.read_file(path)

@st.cache_data
def reproject_raster(in_path, target_crs_str):
    target_crs = rasterio.crs.CRS.from_string(target_crs_str)
    with rasterio.open(in_path) as src:
        if src.crs == target_crs: return in_path 
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height, "compress": 'lzw', "tiled": True})
        reproj_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with rasterio.open(reproj_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=target_crs,
                          resampling=Resampling.nearest)
    return reproj_path

@st.cache_data
def resample_raster(in_path, target_res=10.0): 
    with rasterio.open(in_path) as src:
        if src.res[0] >= target_res: return in_path
        new_width = max(int((src.bounds.right - src.bounds.left) / target_res), 1)
        new_height = max(int((src.bounds.top - src.bounds.bottom) / target_res), 1)
        new_transform = from_bounds(*src.bounds, new_width, new_height)
        kwargs = src.meta.copy()
        kwargs.update({'transform': new_transform, 'width': new_width, 'height': new_height, 'compress': 'lzw'})
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs, dst_transform=new_transform, dst_crs=src.crs,
                          resampling=Resampling.bilinear)
    return out_path

def add_cartographic_elements(ax, crs_is_metric, title):
    ax.set_title(title, pad=20, fontsize=12, weight='bold')
    if crs_is_metric:
        scalebar = ScaleBar(1, "m", length_fraction=0.2, location="lower right")
        ax.add_artist(scalebar)
    ax.text(0.02, 0.98, 'N\n↑', transform=ax.transAxes, fontsize=14, weight='bold', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

def parse_scene_name(filename):
    match = re.search(r'^(\d{4}-\d{2}-\d{2})_([^_]+)', filename)
    if match: return f"{match.group(2).replace('-', ' ').title()} ({match.group(1)})"
    return os.path.splitext(filename)[0][:15]

def inicializar_base(uas_file, sat_file, master_crs, master_gdf, col_clase):
    data = {}
    data['has_uas'] = uas_file is not None
    data['has_sat'] = sat_file is not None
    
    base_file = uas_file if uas_file else sat_file
    t_base = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    t_base.write(base_file.getvalue()); t_base.close()
    
    base_reproj = reproject_raster(t_base.name, master_crs.to_string())
    data['uas_path_10m'] = resample_raster(base_reproj, target_res=10.0)
    
    if data['has_uas']:
        data['uas_path_1m'] = resample_raster(base_reproj, target_res=1.0)
    else:
        data['uas_path_1m'] = data['uas_path_10m']

    if data['has_sat']:
        if data['has_uas']:
            t_sat = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            t_sat.write(sat_file.getvalue()); t_sat.close()
            data['sat_clip_path'] = reproject_raster(t_sat.name, master_crs.to_string())
        else:
            data['sat_clip_path'] = data['uas_path_10m']
            
    if master_gdf is not None and col_clase:
        with rasterio.open(data['uas_path_10m']) as src:
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

def calcular_firmas(data_dict, col_clase, sat_scale, u_idx_list, s_idx_list, sat_name):
    if data_dict['gdf_diss'] is None: return pd.DataFrame(), pd.DataFrame()
    resultados, datos_correlacion = [], []
    u_b, u_g, u_r, u_re, u_n = u_idx_list
    s_b, s_g, s_r, s_re, s_n = s_idx_list
    
    with rasterio.open(data_dict['uas_path_10m']) as uas:
        sat_src = rasterio.open(data_dict['sat_clip_path']) if data_dict['has_sat'] else None
        for _, row in data_dict['gdf_diss'].iterrows():
            pts, bbox, intentos = [], row.geometry.bounds, 0
            while len(pts) < 50 and intentos < 1000:
                p = Point(random.uniform(bbox[0], bbox[2]), random.uniform(bbox[1], bbox[3]))
                if p.within(row.geometry): pts.append(p)
                intentos += 1
            if not pts: continue
            coords = [(pt.x, pt.y) for pt in pts]
            m_uas = np.array(list(uas.sample(coords))).astype(float)
            m_uas[m_uas <= 0] = np.nan
            if sat_src:
                m_sat = np.array(list(sat_src.sample(coords))).astype(float) / sat_scale
                m_sat[m_sat <= 0] = np.nan
                mask = ~np.isnan(m_uas).any(axis=1) & ~np.isnan(m_sat).any(axis=1)
                m_uas, m_sat = m_uas[mask], m_sat[mask]
                if len(m_uas) > 0:
                    f_sat = np.nanmean(m_sat, axis=0)
                    b_map = {s_b:"Azul", s_g:"Verde", s_r:"Rojo", s_re:"Red Edge", s_n:"NIR"}
                    for b in range(sat_src.count):
                        if (b+1) in b_map: resultados.append({'Cobertura': row[col_clase], 'Banda': b_map[b+1], 'Sensor': sat_name, 'Reflectancia': f_sat[b]})
                    for ub, sb, nb in [(u_b, s_b, "Azul"), (u_g, s_g, "Verde"), (u_r, s_r, "Rojo"), (u_re, s_re, "Red Edge"), (u_n, s_n, "NIR")]:
                        if ub > 0 and sb > 0:
                            for uv, sv in zip(m_uas[:, ub-1], m_sat[:, sb-1]): datos_correlacion.append({'Cobertura': row[col_clase], 'Banda': nb, 'UAS': uv, 'SAT': sv})
            if data_dict['has_uas']:
                f_uas = np.nanmean(m_uas, axis=0)
                b_map_u = {u_b:"Azul", u_g:"Verde", u_r:"Rojo", u_re:"Red Edge", u_n:"NIR"}
                for b in range(uas.count):
                    if (b+1) in b_map_u: resultados.append({'Cobertura': row[col_clase], 'Banda': b_map_u[b+1], 'Sensor': 'UAS', 'Reflectancia': f_uas[b]})
        if sat_src: sat_src.close()
    return pd.DataFrame(resultados), pd.DataFrame(datos_correlacion)

def pre_generar_plotly(df_f, df_c, sat_name):
    pf, pc = {}, {}
    if df_f.empty: return pf, pc
    for cob in df_f['Cobertura'].unique():
        sub_f = df_f[df_f['Cobertura'] == cob]
        fig = px.line(sub_f, x="Banda", y="Reflectancia", color="Sensor", markers=True, title=cob)
        fig.update_xaxes(categoryorder='array', categoryarray=["Azul", "Verde", "Rojo", "Red Edge", "NIR"])
        pf[cob] = fig
        if not df_c.empty and 'Cobertura' in df_c.columns:
            sub_c = df_c[df_c['Cobertura'] == cob]
            if len(sub_c) > 5:
                mod = LinearRegression().fit(sub_c[['UAS']], sub_c['SAT'])
                r2 = r2_score(sub_c['SAT'], mod.predict(sub_c[['UAS']]))
                fig_c = px.scatter(sub_c, x="UAS", y="SAT", color="Banda", title=f"{cob} (R²={r2:.3f})")
                pc[cob] = fig_c
    return pf, pc

def generar_mapa_crudo(d, sensor, modo, u_list, s_list, scale, esc_name):
    is_sat = (sensor == "Satelite")
    u_b, u_g, u_r, u_re, u_n = u_list
    s_b, s_g, s_r, s_re, s_n = s_list
    raster_to_open = d['sat_clip_path'] if is_sat else d['uas_path_1m']
    
    with rasterio.open(raster_to_open) as base:
        ext = [base.bounds.left, base.bounds.right, base.bounds.bottom, base.bounds.top]
        def get_b(idx):
            out = np.full((base.height, base.width), np.nan, dtype=np.float32)
            if idx > 0 and idx <= base.count: 
                out = base.read(int(idx)).astype(float)
                if is_sat: out /= scale
            out[out <= 0] = np.nan; return out
        def norm(a):
            p2, p98 = np.nanpercentile(a, [2, 98])
            return (np.clip(a, p2, p98) - p2) / (p98 - p2 + 1e-6)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if modo == "NDVI":
            nir = get_b(s_n if is_sat else u_n)
            red = get_b(s_r if is_sat else u_r)
            ndvi = (nir - red) / (nir + red + 1e-6)
            im = ax.imshow(ndvi, cmap='RdYlGn', extent=ext, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label="NDVI")
            del nir, red, ndvi
        elif "RGB" in modo:
            r = norm(get_b(s_r if is_sat else u_r))
            g = norm(get_b(s_g if is_sat else u_g))
            b = norm(get_b(s_b if is_sat else u_b))
            ax.imshow(np.dstack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)]), extent=ext)
            del r, g, b
            
        add_cartographic_elements(ax, True, f"{modo} - {sensor}")
        try: cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=base.crs.to_string(), alpha=0.5)
        except: pass
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig); gc.collect()
        return buf.getvalue()

def generar_todos_pre_mapas(d, scale, u_l, s_l, name):
    m = {}
    sens = []
    if d['has_uas']: sens.append("UAS")
    if d['has_sat']: sens.append("Satelite")
    for s in sens:
        for modo in ["RGB (Color Real)", "NDVI"]:
            m[f"{s}_{modo}"] = generar_mapa_crudo(d, s, modo, u_l, s_l, scale, name)
    return m

# -----------------------------
# 3. SIDEBAR 
# -----------------------------
with st.sidebar:
    st.header("Configuración del Análisis")
    with st.expander("Archivo Vectorial Global", expanded=True):
        vector_file = st.file_uploader("Subir Vector Opcional (ZIP o GPKG)", type=["zip", "gpkg"])
        if vector_file:
            preview_gdf = load_vector_preview(vector_file)
            st.session_state.raw_gdf = preview_gdf
            resumen_columnas = [{"Columna": c, "Ejemplos": ", ".join(map(str, preview_gdf[c].dropna().unique()[:3]))} for c in preview_gdf.columns if c != 'geometry']
            st.dataframe(pd.DataFrame(resumen_columnas), hide_index=True)
            st.session_state.col_clase = st.selectbox("Seleccionar Columna Clase:", [c for c in preview_gdf.columns if c != 'geometry'], key='col_clase')
        else:
            st.session_state.raw_gdf = None
            st.session_state.col_clase = None

    st.divider()
    st.markdown("**Gestión de Escenas**")
    num_escenas = st.number_input("Cantidad de Escenas Multitemporales", 1, 10, 1)
    archivos_escenas = []
    for i in range(1, num_escenas + 1):
        with st.expander(f"Archivos Escena {i}", expanded=(i==1)):
            archivos_escenas.append({
                "id": i, 
                "uas": st.file_uploader(f"Raster UAS (E{i})", type=["tif"]), 
                "sat": st.file_uploader(f"Raster SAT (E{i})", type=["tif"])
            })

    st.divider()
    with st.expander("Configuración de Bandas y Sensores"):
        sat_name = st.text_input("Nombre Satélite", "Sentinel-2")
        sat_scale = st.number_input("Factor Escala Satélite", 10000.0)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Índices UAS**")
            u_b = st.number_input("Azul", 1); u_g = st.number_input("Verde", 2); u_r = st.number_input("Rojo", 3)
            u_re = st.number_input("Red Edge", 4); u_n = st.number_input("NIR", 5)
        with c2:
            st.markdown("**Índices SAT**")
            s_b = st.number_input("Azul ", 1); s_g = st.number_input("Verde ", 2); s_r = st.number_input("Rojo ", 3)
            s_re = st.number_input("Red Edge ", 0); s_n = st.number_input("NIR ", 4)
        u_idx_list = [u_b, u_g, u_r, u_re, u_n]
        s_idx_list = [s_b, s_g, s_r, s_re, s_n]

    if st.button("Ejecutar Análisis", use_container_width=True):
        first_valid_raster = None
        for e in archivos_escenas:
            if e['uas'] and check_size(e['uas']): first_valid_raster = e['uas']; break
            if e['sat'] and check_size(e['sat']): first_valid_raster = e['sat']; break
            
        if first_valid_raster:
            with MemoryFile(first_valid_raster.getvalue()) as mem: master_crs = mem.open().crs
            master_gdf = st.session_state.raw_gdf.to_crs(master_crs) if st.session_state.raw_gdf is not None else None
            
            st.session_state.data_escenas = {}
            for e in archivos_escenas:
                if (e['uas'] and check_size(e['uas'])) or (e['sat'] and check_size(e['sat'])):
                    name = parse_scene_name(e['uas'].name if e['uas'] else e['sat'].name)
                    st.session_state.data_escenas[name] = inicializar_base(e['uas'], e['sat'], master_crs, master_gdf, st.session_state.col_clase)
            st.session_state.analisis_listo = True
        else:
            st.error("Sube al menos un archivo raster válido y menor a 100MB.")

    if st.button("Reiniciar Entorno"): st.session_state.clear(); st.rerun()

# -----------------------------
# 4. RENDERIZADO PRINCIPAL
# -----------------------------
if st.session_state.get("analisis_listo"):
    names = list(st.session_state.data_escenas.keys())
    tabs = st.tabs([f"Análisis {n}" for n in names])
    
    for idx, name in enumerate(names):
        with tabs[idx]:
            d = st.session_state.data_escenas[name]
            if 'pre_m' not in d:
                with st.spinner(f"Procesando {name}..."):
                    st.image(DOG_GIF_URL, width=200)
                    df_f, df_c = calcular_firmas(d, st.session_state.col_clase, sat_scale, u_idx_list, s_idx_list, sat_name)
                    d['df_f'], d['df_c'] = df_f, df_c
                    d['pf'], d['pc'] = pre_generar_plotly(df_f, df_c, sat_name)
                    d['pre_m'] = generar_todos_pre_mapas(d, sat_scale, u_idx_list, s_idx_list, name)
            
            if d['gdf'] is not None:
                col_m, col_t = st.columns([2, 1])
                with col_m:
                    gdf_map = d['gdf'].to_crs(epsg=4326)
                    m = folium.Map(location=[gdf_map.total_bounds[[1,3]].mean(), gdf_map.total_bounds[[0,2]].mean()], zoom_start=15)
                    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)
                    folium.GeoJson(gdf_map).add_to(m); st_folium(m, width=700, height=400, key=f"f_{name}")
                with col_t:
                    st.plotly_chart(px.pie(d['gdf_diss'], values='area_m2', names=st.session_state.col_clase, title="Distribución de Zonas"), use_container_width=True)
            else:
                st.info("Visualizando imagen completa. Sube un archivo vectorial si deseas analizar por zonas.")
            
            st.divider()
            t_sub = st.tabs(["Cartografía", "Espectros", "Summary Escena"])
            with t_sub[0]:
                sens = []
                if d['has_uas']: sens.append("UAS")
                if d['has_sat']: sens.append("Satelite")
                for s in sens:
                    st.markdown(f"**Sensor: {s}**")
                    cols = st.columns(2)
                    cols[0].image(d['pre_m'][f"{s}_RGB (Color Real)"], use_container_width=True, caption="Color Real")
                    cols[1].image(d['pre_m'][f"{s}_NDVI"], use_container_width=True, caption="NDVI")
            with t_sub[1]:
                if not d['df_f'].empty:
                    cols = st.columns(3)
                    for i, c in enumerate(d['df_f']['Cobertura'].unique()):
                        cols[i%3].plotly_chart(d['pf'][c], use_container_width=True)
                else: st.write("Requiere archivo vectorial para extraer firmas.")
            with t_sub[2]:
                if d['has_sat'] and not d['df_c'].empty:
                    cols = st.columns(3)
                    for i, c in enumerate(d['pc']):
                        cols[i%3].plotly_chart(d['pc'][c], use_container_width=True)
                else: st.write("Requiere archivo vectorial e imágenes de ambos sensores (UAS y SAT) para el análisis de correlación.")
