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
# 1. FUNCIONES DE PROCESAMIENTO
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
    t_uas_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    t_uas_raw.write(uas_file.getvalue()); t_uas_raw.close()
    uas_reproj = reproject_raster(t_uas_raw.name, master_crs.to_string())
    data['uas_path_1m'] = resample_raster(uas_reproj, target_res=1.0)
    data['uas_path_10m'] = resample_raster(uas_reproj, target_res=10.0)
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
    else: data['has_sat'] = False
    return data

# -----------------------------
# 3. FUNCIONES DE ANÁLISIS
# -----------------------------
def calcular_firmas(data_dict, col_clase, sat_scale, u_idx_list, s_idx_list, sat_name):
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

def generar_mapa_crudo(d, sensor, modo, u_list, s_list, scale, esc_name, b_sel=1):
    is_sat = (sensor == "Satelite")
    u_b, u_g, u_r, u_re, u_n = u_list
    s_b, s_g, s_r, s_re, s_n = s_list
    with rasterio.open(d['uas_path_1m']) as base:
        ext = [base.bounds.left, base.bounds.right, base.bounds.bottom, base.bounds.top]
        def get_b(u_idx, s_idx):
            out = np.full((base.height, base.width), np.nan, dtype=np.float32)
            if not is_sat: out = base.read(int(u_idx)).astype(float) if u_idx > 0 else out
            else:
                with rasterio.open(d['sat_clip_path']) as ss:
                    if s_idx > 0: reproject(rasterio.band(ss, int(s_idx)), out, src_transform=ss.transform, src_crs=ss.crs, dst_transform=base.transform, dst_crs=base.crs, resampling=Resampling.bilinear)
                    out /= scale
            out[out <= 0] = np.nan; return out
        def norm(a):
            p2, p98 = np.nanpercentile(a, [2, 98])
            return (np.clip(a, p2, p98) - p2) / (p98 - p2 + 1e-6)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if modo == "NDVI":
            nir, red = get_b(u_n, s_n), get_b(u_r, s_r)
            ndvi = (nir - red) / (nir + red + 1e-6)
            im = ax.imshow(ndvi, cmap='RdYlGn', extent=ext, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
        elif "RGB" in modo:
            r, g, b = norm(get_b(u_r, s_r)), norm(get_b(u_g, s_g)), norm(get_b(u_b, s_b))
            ax.imshow(np.dstack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)]), extent=ext)
        add_cartographic_elements(ax, True, f"{modo} - {sensor}"); buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); return buf.getvalue()

def generar_todos_pre_mapas(d, scale, u_l, s_l, name):
    m = {}
    sens = ["UAS", "Satelite"] if d['has_sat'] else ["UAS"]
    for s in sens:
        for modo in ["RGB (Color Real)", "NDVI"]:
            m[f"{s}_{modo}"] = generar_mapa_crudo(d, s, modo, u_l, s_l, scale, name)
    return m

# -----------------------------
# 4. SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Configuración")
    v_file = st.file_uploader("Vector", type=["zip", "gpkg"])
    if v_file:
        st.session_state.raw_gdf = load_vector_preview(v_file)
        # GUARDAR COLUMNA EN SESSION STATE
        st.session_state.col_clase = st.selectbox("Clase:", [c for c in st.session_state.raw_gdf.columns if c != 'geometry'])
    
    n_esc = st.number_input("Escenas", 1, 10, 1)
    esc_files = []
    for i in range(1, n_esc + 1):
        with st.expander(f"Escena {i}"):
            esc_files.append({"u": st.file_uploader(f"UAS {i}", type=["tif"]), "s": st.file_uploader(f"SAT {i}", type=["tif"])})
    s_name = st.text_input("Satélite", "Sentinel-2"); s_scale = st.number_input("Escala", 10000.0)
    c1, c2 = st.columns(2)
    with c1: u_l = [st.number_input(f"U-{b}", i+1) for i, b in enumerate(["B","G","R","RE","N"])]
    with c2: s_l = [st.number_input(f"S-{b}", i+1) for i, b in enumerate(["B","G","R","RE","N"])]
    
    if st.button("Ejecutar Análisis", use_container_width=True):
        with MemoryFile(esc_files[0]['u'].getvalue()) as mem: m_crs = mem.open().crs
        st.session_state.data_escenas = {}
        for e in esc_files:
            if e['u']:
                name = parse_scene_name(e['u'].name)
                st.session_state.data_escenas[name] = inicializar_base(e['u'], e['s'], m_crs, st.session_state.raw_gdf.to_crs(m_crs), st.session_state.col_clase)
        st.session_state.analisis_listo = True
    if st.button("Reiniciar"): st.session_state.clear(); st.rerun()

# -----------------------------
# 5. RENDERIZADO PRINCIPAL
# -----------------------------
if st.session_state.get("analisis_listo"):
    names = list(st.session_state.data_escenas.keys())
    tabs = st.tabs([f"Análisis {n}" for n in names] + ["Comparación Global"])
    
    for idx, name in enumerate(names):
        with tabs[idx]:
            d = st.session_state.data_escenas[name]
            if 'pre_m' not in d:
                with st.spinner(f"Procesando {name}..."):
                    st.image(DOG_GIF_URL, width=200)
                    # USAR VARIABLE DESDE SESSION STATE
                    df_f, df_c = calcular_firmas(d, st.session_state.col_clase, s_scale, u_l, s_l, s_name)
                    d['df_f'], d['df_c'] = df_f, df_c
                    d['pf'], d['pc'] = pre_generar_plotly(df_f, df_c, s_name)
                    d['pre_m'] = generar_todos_pre_mapas(d, s_scale, u_l, s_l, name)
            
            col_m, col_t = st.columns([2, 1])
            with col_m:
                gdf_map = d['gdf'].to_crs(epsg=4326)
                m = folium.Map(location=[gdf_map.total_bounds[[1,3]].mean(), gdf_map.total_bounds[[0,2]].mean()], zoom_start=15)
                folium.GeoJson(gdf_map).add_to(m); st_folium(m, width=700, height=400, key=f"f_{name}")
            with col_t:
                # CORRECCIÓN NAMEERROR: USAR SESSION STATE
                st.plotly_chart(px.pie(d['gdf_diss'], values='area_m2', names=st.session_state.col_clase), use_container_width=True)
            
            st.divider()
            t_sub = st.tabs(["Cartografía", "Espectros", "Summary"])
            with t_sub[0]:
                for s in (["UAS", "Satelite"] if d['has_sat'] else ["UAS"]):
                    st.markdown(f"**Sensor: {s}**")
                    cols = st.columns(2)
                    cols[0].image(d['pre_m'][f"{s}_RGB (Color Real)"], use_container_width=True)
                    cols[1].image(d['pre_m'][f"{s}_NDVI"], use_container_width=True)
            with t_sub[1]:
                cols = st.columns(3)
                for i, c in enumerate(d['df_f']['Cobertura'].unique()):
                    cols[i%3].plotly_chart(d['pf'][c], use_container_width=True)
            with t_sub[2]:
                if d['has_sat'] and not d['df_c'].empty:
                    cols = st.columns(3)
                    for i, c in enumerate(d['pc']):
                        cols[i%3].plotly_chart(d['pc'][c], use_container_width=True)

    # --- PESTAÑA COMPARACIÓN GLOBAL ---
    with tabs[-1]:
        st.header("Análisis Comparativo Global")
        all_f = pd.concat([st.session_state.data_escenas[n]['df_f'].assign(Escena=n) for n in names])
        cobs = all_f['Cobertura'].unique()
        
        gt1, gt2, gt3 = st.tabs(["Evolución UAS", f"Evolución {s_name}", "Resumen R²"])
        
        with gt1:
            cols = st.columns(3)
            for i, c in enumerate(cobs):
                sub = all_f[(all_f['Cobertura']==c) & (all_f['Sensor']=='UAS')]
                fig = px.line(sub, x="Banda", y="Reflectancia", color="Escena", markers=True, title=f"UAS: {c}")
                fig.update_xaxes(categoryorder='array', categoryarray=["Azul", "Verde", "Rojo", "Red Edge", "NIR"])
                cols[i%3].plotly_chart(fig, use_container_width=True)
        
        with gt2:
            cols = st.columns(3)
            for i, c in enumerate(cobs):
                sub = all_f[(all_f['Cobertura']==c) & (all_f['Sensor']==s_name)]
                if not sub.empty:
                    fig = px.line(sub, x="Banda", y="Reflectancia", color="Escena", markers=True, title=f"Sat: {c}")
                    fig.update_xaxes(categoryorder='array', categoryarray=["Azul", "Verde", "Rojo", "Red Edge", "NIR"])
                    cols[i%3].plotly_chart(fig, use_container_width=True)
        
        with gt3:
            all_c = pd.concat([st.session_state.data_escenas[n]['df_c'] for n in names if not st.session_state.data_escenas[n]['df_c'].empty])
            if not all_c.empty:
                r2_results = []
                for c in cobs:
                    sub_c = all_c[all_c['Cobertura'] == c]
                    if len(sub_c) > 5:
                        mod = LinearRegression().fit(sub_c[['UAS']], sub_c['SAT'])
                        r2_val = r2_score(sub_c['SAT'], mod.predict(sub_c[['UAS']]))
                        r2_results.append({'Cobertura': c, 'R2': r2_val})
                
                if r2_results:
                    df_r2 = pd.DataFrame(r2_results)
                    st.plotly_chart(px.bar(df_r2, x='Cobertura', y='R2', color='R2', title="Ajuste Radiométrico Global"), use_container_width=True)
                    
                    st.markdown("### Regresiones Consolidadas")
                    cols = st.columns(3)
                    for i, c in enumerate(cobs):
                        sub_c = all_c[all_c['Cobertura'] == c]
                        if not sub_c.empty:
                            mod = LinearRegression().fit(sub_c[['UAS']], sub_c['SAT'])
                            fig = px.scatter(sub_c, x="UAS", y="SAT", color="Banda", title=f"{c} (Global)")
                            fig.add_trace(go.Scatter(x=[sub_c['UAS'].min(), sub_c['UAS'].max()], y=mod.predict([[sub_c['UAS'].min()], [sub_c['UAS'].max()]]), mode='lines', name='Tendencia', line=dict(color='black', dash='dot')))
                            cols[i%3].plotly_chart(fig, use_container_width=True)