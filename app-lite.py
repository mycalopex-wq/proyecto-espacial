if vector_file:
        preview_gdf = load_vector_preview(vector_file)
        st.session_state.raw_gdf = preview_gdf
        
        # Guardamos la selección en una variable temporal primero
        columna_elegida = st.selectbox("Columna Clase:", [c for c in preview_gdf.columns if c != 'geometry'])
        # Luego la asignamos a la memoria de la sesión
        st.session_state.col_clase = columna_elegida
    else:
        st.session_state.raw_gdf = None
        st.session_state.col_clase = None
