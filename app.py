# -*- coding: utf-8 -*-
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.mask import mask
import tempfile, zipfile, os, re, io
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
# 1. FUNCIONES PRINCIPALES
# -----------------------------
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
        kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height, "compress": 'lzw', "tiled": True, "num_threads": -1})
        reproj_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with rasterio.open(reproj_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=target_crs,
                          resampling=Resampling.nearest, num_threads=-1)
    return reproj_path

@st.cache_data
def resample_raster(in_path, target_res=1.0): 
    with rasterio.open(in_path) as src:
        if src.res[0] >= target_res: return in_path
        new_width = int((src.bounds.right - src.bounds.left) / target_res)
        new_height = int((src.bounds.top - src.bounds.bottom) / target_res)
        new_transform = from_bounds(*src.bounds, new_width, new_height)
        kwargs = src.meta.copy()
        kwargs.update({'transform': new_transform, 'width': new_width, 'height': new_height, 'compress': 'lzw', 'tiled': True})
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs, dst_transform=new_transform, dst_crs=src.crs,
                          resampling=Resampling.bilinear, num_threads=-1)
    return out_path

def add_cartographic_elements(ax, crs_is_metric, title):
    ax.set_title(title, pad=20, fontsize=14, color='black', weight='bold')
    ax.tick_params(axis='both', colors='black', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.2)
    ax.set_xlabel('Este (X)', color='black', fontsize=10)
    ax.set_ylabel('Norte (Y)', color='black', fontsize=10)
    if crs_is_metric:
        scalebar = ScaleBar(1, "m", length_fraction=0.2, location="lower right", color="black", box_color="white", box_alpha=0.8)
        ax.add_artist(scalebar)
    ax.text(0.05, 0.95, 'N\n↑', transform=ax.transAxes, color='black', fontsize=16, ha='center', va='center', weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

def create_context_maps(ax_regional, ax_national, main_gdf):
    gdf_wm = main_gdf.to_crs(epsg=3857)
    gdf_wm.plot(ax=ax_regional, facecolor='none', edgecolor='red', linewidth=2)
    try:
        minx, miny, maxx, maxy = gdf_wm.total_bounds
        cx_coord, cy_coord = (minx + maxx) / 2, (miny + maxy) / 2
        buffer_reg_x, buffer_reg_y = 4000, 12000  
        ax_regional.set_xlim(cx_coord - buffer_reg_x, cx_coord + buffer_reg_x)
        ax_regional.set_ylim(cy_coord - buffer_reg_y, cy_coord + buffer_reg_y)
        cx.add_basemap(ax_regional, source=cx.providers.OpenStreetMap.Mapnik, zoom=12, alpha=0.7)
    except Exception: pass 
    centroid = gdf_wm.centroid
    centroid.plot(ax=ax_national, color='red', marker='*', markersize=300, edgecolor='black', linewidth=1.5, zorder=5)
    try:
        buffer_nat_x, buffer_nat_y = 250000, 750000 
        ax_national.set_xlim(cx_coord - buffer_nat_x, cx_coord + buffer_nat_x)
        ax_national.set_ylim(cy_coord - buffer_nat_y, cy_coord + buffer_nat_y)
        cx.add_basemap(ax_national, source=cx.providers.CartoDB.Positron, zoom=5, alpha=0.9)
    except Exception: pass
    for ax_map, title in zip([ax_regional, ax_national], ["Contexto Regional", "Ubicación País"]):
        ax_map.set_xticks([]); ax_map.set_yticks([])
        for spine in ax_map.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(1.5)
        ax_map.set_title(title, fontsize=11, weight='bold', color='black', pad=10)

def parse_scene_name(filename):
    match = re.search(r'^(\d{4}-\d{2}-\d{2})_([^_]+)', filename)
    if match:
        fecha = match.group(1)
        lugar = match.group(2).replace('-', ' ').title()
        return f"{lugar} ({fecha})"
    return os.path.splitext(filename)[0][:15]

# -----------------------------
# 2. MOTOR DE PROCESAMIENTO
# -----------------------------
def inicializar_base(uas_file, sat_file, master_crs, master_gdf, col_clase):
    data = {}
    t_uas_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    t_uas_raw.write(uas_file.getvalue()); t_uas_raw.close()
    t_uas_raw_name = reproject_raster(t_uas_raw.name, master_crs.to_string())
    data['uas_path_1m'] = resample_raster(t_uas_raw_name, target_res=1.0)
    data['uas_path_10m'] = resample_raster(t_uas_raw_name, target_res=10.0)
    with rasterio.open(data['uas_path_10m']) as src:
        gdf_cortado = gpd.clip(master_gdf, box(*src.bounds))
        if gdf_cortado.crs.is_geographic: gdf_area = gdf_cortado.to_crs(epsg=3857)
        else: gdf_area = gdf_cortado.copy()
        gdf_cortado['area_m2'] = gdf_area.geometry.area
        data['gdf'] = gdf_cortado
        data['gdf_diss'] = gdf_cortado.dissolve(by=col_clase, aggfunc={'area_m2': 'sum'}).reset_index()
    if sat_file is not None:
        t_sat = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        t_sat.write(sat_file.getvalue()); t_sat.close()
        data['sat_clip_path'] = reproject_raster(t_sat.name, master_crs.to_string())
        data['has_sat'] = True
    else:
        data['has_sat'] = False
    return data

def calcular_firmas(data_dict, col_clase, sat_scale, b_idx, g_idx, r_idx, re_idx, n_idx, s_b_idx, s_g_idx, s_r_idx, s_re_idx, s_n_idx, sat_name):
    resultados, datos_correlacion = [], []
    uas_bands = {name for idx, name in [(b_idx,"Azul"), (g_idx,"Verde"), (r_idx,"Rojo"), (re_idx,"Red Edge"), (n_idx,"NIR")] if idx > 0}
    sat_bands = {name for idx, name in [(s_b_idx,"Azul"), (s_g_idx,"Verde"), (s_r_idx,"Rojo"), (s_re_idx,"Red Edge"), (s_n_idx,"NIR")] if idx > 0}
    bandas_comunes = uas_bands.intersection(sat_bands) if data_dict['has_sat'] else set()
    
    with rasterio.open(data_dict['uas_path_10m']) as uas:
        sat_src = rasterio.open(data_dict['sat_clip_path']) if data_dict['has_sat'] else None
        for _, row in data_dict['gdf_diss'].iterrows():
            pts, bbox, intentos = [], row.geometry.bounds, 0
            while len(pts) < 100 and intentos < 2000:
                p = Point(random.uniform(bbox[0], bbox[2]), random.uniform(bbox[1], bbox[3]))
                if p.within(row.geometry): pts.append(p)
                intentos += 1
            if not pts: continue
            coordenadas = [(pt.x, pt.y) for pt in pts]
            muestras_uas = np.array(list(uas.sample(coordenadas))).astype(float)
            muestras_uas[muestras_uas <= 0] = np.nan
            
            if sat_src:
                muestras_sat = np.array(list(sat_src.sample(coordenadas))).astype(float) / sat_scale
                muestras_sat[muestras_sat <= 0] = np.nan
                mask_ambos = ~np.isnan(muestras_uas).any(axis=1) & ~np.isnan(muestras_sat).any(axis=1)
                muestras_uas, muestras_sat = muestras_uas[mask_ambos], muestras_sat[mask_ambos]
                if len(muestras_uas) > 0:
                    firma_sat = np.nanmean(muestras_sat, axis=0)
                    band_names_sat_map = {idx: name for idx, name in [(s_b_idx,"Azul"), (s_g_idx,"Verde"), (s_r_idx,"Rojo"), (s_re_idx,"Red Edge"), (s_n_idx,"NIR")] if idx > 0}
                    for b in range(sat_src.count):
                        if (b+1) in band_names_sat_map: resultados.append({'Cobertura': row[col_clase], 'Banda': band_names_sat_map[b+1], 'Sensor': sat_name, 'Reflectancia': firma_sat[b]})
                    for nb in bandas_comunes:
                        u_idx = {n: i for i, n in [(b_idx,"Azul"), (g_idx,"Verde"), (r_idx,"Rojo"), (re_idx,"Red Edge"), (n_idx,"NIR")] if i > 0}[nb] - 1
                        s_idx = {n: i for i, n in [(s_b_idx,"Azul"), (s_g_idx,"Verde"), (s_r_idx,"Rojo"), (s_re_idx,"Red Edge"), (s_n_idx,"NIR")] if i > 0}[nb] - 1
                        if u_idx < muestras_uas.shape[1] and s_idx < muestras_sat.shape[1]:
                            for uv, sv in zip(muestras_uas[:, u_idx], muestras_sat[:, s_idx]): datos_correlacion.append({'Cobertura': row[col_clase], 'Banda': nb, 'UAS': uv, 'SAT': sv})
            
            firma_uas = np.nanmean(muestras_uas, axis=0)
            band_names_uas_map = {idx: name for idx, name in [(b_idx,"Azul"), (g_idx,"Verde"), (r_idx,"Rojo"), (re_idx,"Red Edge"), (n_idx,"NIR")] if idx > 0}
            for b in range(uas.count):
                if (b+1) in band_names_uas_map: resultados.append({'Cobertura': row[col_clase], 'Banda': band_names_uas_map[b+1], 'Sensor': 'UAS', 'Reflectancia': firma_uas[b]})
        if sat_src: sat_src.close()
    return pd.DataFrame(resultados), pd.DataFrame(datos_correlacion)

def pre_generar_plotly(df_firmas, df_corr, sat_name):
    pre_firmas, pre_corr = {}, {}
    if df_firmas.empty: return pre_firmas, pre_corr
    
    coberturas = df_firmas['Cobertura'].unique()
    for cob in coberturas:
        df_f = df_firmas[df_firmas['Cobertura'] == cob]
        fig_f = px.line(df_f, x="Banda", y="Reflectancia", color="Sensor", markers=True, title=f"{cob}", color_discrete_map={'UAS':'blue', sat_name:'red'})
        fig_f.update_traces(line=dict(width=3, dash='dash'), marker=dict(size=7, symbol='circle-dot'))
        fig_f.update_xaxes(categoryorder='array', categoryarray=["Azul", "Verde", "Rojo", "Red Edge", "NIR"])
        fig_f.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, title=None), margin=dict(l=10, r=10, t=40, b=90))
        pre_firmas[cob] = fig_f
        
        if not df_corr.empty and 'Cobertura' in df_corr.columns:
            df_c = df_corr[df_corr['Cobertura'] == cob]
            if not df_c.empty and len(df_c) > 5:
                X, y = df_c['UAS'].values.reshape(-1, 1), df_c['SAT'].values
                mod = LinearRegression().fit(X, y); r2 = r2_score(y, mod.predict(X))
                fig_c = px.scatter(df_c, x="UAS", y="SAT", color="Banda", title=f"{cob} (R²={r2:.3f})")
                x_min, x_max = X.min(), X.max()
                fig_c.add_trace(go.Scatter(x=[x_min, x_max], y=mod.predict([[x_min], [x_max]]), mode='lines', name='Tendencia', line=dict(color='black', width=2, dash='dot')))
                fig_c.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, title=None), margin=dict(l=10, r=10, t=40, b=90))
                pre_corr[cob] = fig_c
    return pre_firmas, pre_corr

def generar_mapa_crudo(data_dict, sensor_sel, vis_mode, b_idx, g_idx, r_idx, re_idx, n_idx, s_b_idx, s_g_idx, s_r_idx, s_re_idx, s_n_idx, sat_scale, escena_name, banda_sel=1):
    is_sat = (sensor_sel == "Satelite")
    with rasterio.open(data_dict['uas_path_1m']) as base_src:
        ext = [base_src.bounds.left, base_src.bounds.right, base_src.bounds.bottom, base_src.bounds.top]
        uas_data = base_src.read()
        master_mask = (uas_data <= 0).all(axis=0)
        def obt_banda(idx_u, idx_s):
            out = np.full((base_src.height, base_src.width), np.nan, dtype=np.float32)
            if not is_sat: out = base_src.read(int(idx_u)).astype(float) if 0 < idx_u <= base_src.count else out
            else:
                if not data_dict.get('sat_clip_path'): return out
                with rasterio.open(data_dict['sat_clip_path']) as ss:
                    if 0 < idx_s <= ss.count: reproject(rasterio.band(ss, int(idx_s)), out, src_transform=ss.transform, src_crs=ss.crs, dst_transform=base_src.transform, dst_crs=base_src.crs, resampling=Resampling.bilinear)
                    out /= sat_scale
            out[master_mask] = np.nan; out[out <= 0] = np.nan; return out
        def norm(arr):
            if np.isnan(arr).all(): return arr
            p2, p98 = np.nanpercentile(arr, [2, 98])
            return (np.clip(arr, p2, p98) - p2) / (p98 - p2 + 1e-6)
        fig = plt.figure(figsize=(12, 5.5), dpi=150); gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1])
        ax, axr, axn = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        if vis_mode == "NDVI":
            rn, rr = obt_banda(n_idx, s_n_idx), obt_banda(r_idx, s_r_idx)
            ndvi = (rn - rr) / (rn + rr + 1e-6); p2, p98 = np.nanpercentile(ndvi, [2, 98])
            im = ax.imshow(ndvi, cmap='RdYlGn', vmin=p2, vmax=p98, extent=ext, interpolation='bicubic')
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label('NDVI')
        elif "Color" in vis_mode or "RGB" in vis_mode:
            c1 = norm(obt_banda(n_idx, s_n_idx) if "Falso" in vis_mode else obt_banda(r_idx, s_r_idx))
            c2 = norm(obt_banda(r_idx, s_r_idx) if "Falso" in vis_mode else obt_banda(g_idx, s_g_idx))
            c3 = norm(obt_banda(g_idx, s_g_idx) if "Falso" in vis_mode else obt_banda(b_idx, s_b_idx))
            ax.imshow(np.dstack([np.nan_to_num(c1, nan=1.0), np.nan_to_num(c2, nan=1.0), np.nan_to_num(c3, nan=1.0), np.where(np.isnan(c1), 0, 1)]), extent=ext, interpolation='bicubic')
        elif "Banda Pura" in vis_mode:
            b_norm = norm(obt_banda(banda_sel, banda_sel))
            ax.imshow(b_norm, cmap='gray', extent=ext, interpolation='bicubic')
        add_cartographic_elements(ax, True, f"{vis_mode} - {escena_name}"); create_context_maps(axr, axn, data_dict['gdf'])
        fig.subplots_adjust(left=0.02, right=0.98, wspace=0.1)
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white'); plt.close(fig); return buf.getvalue()

def generar_todos_pre_mapas(data_dict, sat_scale, b_idx, g_idx, r_idx, re_idx, n_idx, s_b_idx, s_g_idx, s_r_idx, s_re_idx, s_n_idx, escena_name):
    pre_mapas = {}
    modos_pre = ["RGB (Color Real)", "Falso Color (NIR-R-G)", "NDVI"]
    sensores_pre = ["UAS"]
    if data_dict.get('has_sat'): sensores_pre.append("Satelite")
    for sensor in sensores_pre:
        for modo in modos_pre:
            pre_mapas[f"{sensor}_{modo}"] = generar_mapa_crudo(data_dict, sensor, modo, b_idx, g_idx, r_idx, re_idx, n_idx, s_b_idx, s_g_idx, s_r_idx, s_re_idx, s_n_idx, sat_scale, escena_name)
    return pre_mapas

# -----------------------------
# 3. INTERFAZ (SIDEBAR COMPLETO RESTAURADO)
# -----------------------------
col_clase_input = None
with st.sidebar:
    st.header("Configuración del Análisis")
    with st.expander("Archivo Vectorial Global", expanded=True):
        vector_file = st.file_uploader("Archivo Vectorial", type=["zip", "gpkg"])
        if vector_file:
            preview_gdf = load_vector_preview(vector_file); st.session_state.raw_gdf = preview_gdf
            resumen_columnas = [{"Columna": c, "Ejemplos": ", ".join(map(str, preview_gdf[c].dropna().unique()[:3]))} for c in preview_gdf.columns if c != 'geometry']
            st.dataframe(pd.DataFrame(resumen_columnas), hide_index=True, width="stretch")
            col_clase_input = st.selectbox("Columna Clase:", [c for c in preview_gdf.columns if c != 'geometry'])

    st.divider()
    st.markdown("**Gestión de Escenas**")
    num_escenas = st.number_input("Cantidad de Escenas", 1, 10, 1)
    archivos_escenas = []
    for i in range(1, num_escenas + 1):
        with st.expander(f"Archivos Escena {i}", expanded=(i==1)):
            archivos_escenas.append({
                "id": i, 
                "uas": st.file_uploader(f"Raster UAS (E{i})", type=["tif"]), 
                "sat": st.file_uploader(f"Raster SAT (E{i})", type=["tif"])
            })

    st.divider()
    with st.expander("Configuración de Sensores y Bandas"):
        sat_name = st.text_input("Nombre del Satélite", "Sentinel-2")
        sat_scale = st.number_input("Factor Escala", 10000.0)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**UAS**")
            u_b = st.number_input("Azul", 1, key="u_b")
            u_g = st.number_input("Verde", 2, key="u_g")
            u_r = st.number_input("Rojo", 3, key="u_r")
            u_re = st.number_input("Red Edge", 4, key="u_re")
            u_n = st.number_input("NIR", 5, key="u_n")
        with c2:
            st.markdown("**SAT**")
            s_b = st.number_input("Azul", 1, key="s_b")
            s_g = st.number_input("Verde", 2, key="s_g")
            s_r = st.number_input("Rojo", 3, key="s_r")
            s_re = st.number_input("Red Edge", 4, key="s_re")
            s_n = st.number_input("NIR", 5, key="s_n")

    st.divider()
    if st.button("Ejecutar Análisis Espacial", width="stretch"):
        if vector_file and archivos_escenas[0]['uas']:
            with st.spinner("Preparando entorno..."):
                with MemoryFile(archivos_escenas[0]['uas'].getvalue()) as mem: master_crs = mem.open().crs
                st.session_state.master_gdf = st.session_state.raw_gdf.to_crs(master_crs); st.session_state.data_escenas = {}
                for e in archivos_escenas:
                    if e['uas']:
                        name = parse_scene_name(e['uas'].name)
                        db = inicializar_base(e['uas'], e['sat'], master_crs, st.session_state.master_gdf, col_clase_input)
                        st.session_state.data_escenas[name] = db
                st.session_state.analisis_listo = True
    if st.button("Reiniciar Entorno"): st.session_state.clear(); st.rerun()

# -----------------------------
# 4. RENDERIZADO
# -----------------------------
if st.session_state.get("analisis_listo"):
    names = list(st.session_state.data_escenas.keys())
    tabs = st.tabs([f"Análisis {n}" for n in names] + (["Comparación Global"] if len(names)>0 else []))
    
    for idx, name in enumerate(names):
        with tabs[idx]:
            d = st.session_state.data_escenas[name]
            st.subheader(f"Escena: {name}")
            col_mapa, col_torta = st.columns([2, 1])
            if 'color_map' not in st.session_state:
                st.session_state.color_map = {c: '#%06x' % random.randint(0, 0xFFFFFF) for c in st.session_state.master_gdf[col_clase_input].unique()}
            with col_mapa:
                gdf_map = d['gdf'].to_crs(epsg=4326)
                m = folium.Map(location=[gdf_map.total_bounds[[1,3]].mean(), gdf_map.total_bounds[[0,2]].mean()], zoom_start=15)
                folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)
                gdf_map["color"] = gdf_map[col_clase_input].map(st.session_state.color_map)
                folium.GeoJson(gdf_map, style_function=lambda f: {'fillColor': f['properties']['color'], 'color': 'white', 'weight': 1, 'fillOpacity': 0.6}).add_to(m)
                st_folium(m, width=800, height=400, returned_objects=[], key=f"folium_{name}")
            with col_torta:
                df_stats = d['gdf_diss'].copy()
                st.metric("Total Hectáreas", f"{df_stats['area_m2'].sum()/10000:.2f} ha")
                fig_pie = px.pie(df_stats, values='area_m2', names=col_clase_input, hole=0.4, color=col_clase_input, color_discrete_map=st.session_state.color_map)
                st.plotly_chart(fig_pie, width="stretch")
            
            if 'pre_m' not in d:
                loading_ph = st.empty()
                with loading_ph.container():
                    st.markdown("<h3 style='text-align: center;'>Procesando escena espacial...</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1,1,1]); col2.image(DOG_GIF_URL, width="stretch")
                    status_text, pbar = st.empty(), st.progress(0)
                    status_text.info("Paso 1/3: Muestreo radiométrico..."); df_f, df_c = calcular_firmas(d, col_clase_input, sat_scale, u_b, u_g, u_r, u_re, u_n, s_b, s_g, s_r, s_re, s_n, sat_name)
                    d['df_firmas'], d['df_corr'] = df_f, df_c; pbar.progress(33)
                    status_text.info("Paso 2/3: Modelando regresiones..."); d['pre_p_f'], d['pre_p_c'] = pre_generar_plotly(df_f, df_c, sat_name); pbar.progress(66)
                    status_text.info("Paso 3/3: Renderizando mapas..."); d['pre_m'] = generar_todos_pre_mapas(d, sat_scale, u_b, u_g, u_r, u_re, u_n, s_b, s_g, s_r, s_re, s_n, name); pbar.progress(100)
                st.session_state.data_escenas[name] = d; loading_ph.empty()

            sub_tabs = st.tabs(["Cartografía", "Análisis por Cobertura", "Summary de Escena"])
            with sub_tabs[0]:
                s_sel = st.tabs(["UAS", "Satelite"]) if d['has_sat'] else [st.container()]
                for i, sensor in enumerate(["UAS", "Satelite"] if d['has_sat'] else ["UAS"]):
                    with s_sel[i]:
                        m_tabs = st.tabs(["RGB", "Falso Color", "NDVI", "Banda Pura"])
                        for j, m in enumerate(["RGB (Color Real)", "Falso Color (NIR-R-G)", "NDVI"]):
                            with m_tabs[j]: st.image(d['pre_m'][f"{sensor}_{m}"], width="stretch")
                        with m_tabs[3]:
                            banda_sel = st.selectbox("Seleccione Banda:", range(1, 6), key=f"bp_{name}_{sensor}")
                            st.image(generar_mapa_crudo(d, sensor, "Banda Pura", u_b, u_g, u_r, u_re, u_n, s_b, s_g, s_r, s_re, s_n, sat_scale, name, banda_sel), width="stretch")
            with sub_tabs[1]:
                cobs = d['df_firmas']['Cobertura'].unique(); cols = st.columns(3)
                for i, c in enumerate(cobs):
                    with cols[i%3]: st.plotly_chart(d['pre_p_f'][c], width="stretch")
            with sub_tabs[2]:
                st.subheader(f"Resumen Radiométrico: {name}")
                if d['has_sat'] and not d['df_corr'].empty:
                    r2_list_escena = []
                    for c in cobs:
                        if 'Cobertura' in d['df_corr'].columns:
                            df_sub = d['df_corr'][d['df_corr']['Cobertura'] == c]
                            if len(df_sub) > 5:
                                mod = LinearRegression().fit(df_sub[['UAS']], df_sub['SAT'])
                                r2_list_escena.append({'Cobertura': c, 'R2': r2_score(df_sub['SAT'], mod.predict(df_sub[['UAS']]))})
                    if r2_list_escena:
                        st.plotly_chart(px.bar(pd.DataFrame(r2_list_escena), x='Cobertura', y='R2', color='R2', title="Ajuste Radiométrico (Escena Actual)"), width="stretch")
                cols = st.columns(3)
                for i, c in enumerate(cobs):
                    if c in d.get('pre_p_c', {}):
                        with cols[i%3]: st.plotly_chart(d['pre_p_c'][c], width="stretch")