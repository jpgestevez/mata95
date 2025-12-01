import streamlit as st
import numpy as np
import numexpr as ne
import pyvista as pv
import tempfile
import os

# ==========================================
# 1. CONFIGURAÇÃO DO SERVIDOR
# ==========================================
try:
    pv.start_xvfb()
except:
    pass

pv.OFF_SCREEN = True 
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'

st.set_page_config(page_title="Copos 3D", layout="wide")

# ==========================================
# 2. FUNÇÕES MATEMÁTICAS
# ==========================================
def make_f_func(expr: str):
    """Interpreta a função digitada pelo usuário."""
    def f(z):
        z_arr = np.asarray(z)
        local_dict = {'z': z_arr, 'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}
        try:
            return ne.evaluate(expr, local_dict=local_dict)
        except:
            return np.zeros_like(z_arr)
    return f

def calcular_volume(r0, H, f_func, n_z=1000):
    """Calcula o volume."""
    Z = np.linspace(0.0, H, n_z)
    try:
        Rz = r0 + f_func(Z)
        Rz = np.maximum(Rz, 0)
        area_layers = np.pi * (Rz**2)
        return np.trapezoid(area_layers, Z)
    except:
        return 0.0

def gerar_mesh(r0, H, f_func):
    """Gera a visualização com fundo conectado corretamente e força formato Superfície."""
    n_z = 150
    n_theta = 100  # Resolução angular
    
    Z = np.linspace(0.0, H, n_z)
    theta = np.linspace(0.0, 2*np.pi, n_theta)
    Tg, Zg = np.meshgrid(theta, Z)
    
    try:
        # Calcula o raio da parede
        Rg = r0 + f_func(Zg)
        Rg = np.maximum(Rg, 0)
        
        # Calcula o raio exato na base para o fundo
        raio_base_real = r0 + f_func(np.array([0.0]))[0]
        raio_base_real = max(raio_base_real, 0)
        
    except:
        return None

    # Coordenadas da Parede
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    pts = np.column_stack((X.ravel(), Y.ravel(), Zg.ravel()))
    
    # Cria a malha da parede
    wall_grid = pv.StructuredGrid()
    wall_grid.dimensions = [n_theta, n_z, 1]
    wall_grid.points = pts
    
    # Converte a parede para superfície
    wall_mesh = wall_grid.extract_surface().triangulate()
    
    # Cria o fundo
    bottom = pv.Circle(radius=raio_base_real, resolution=n_theta)
    
    # Força a triangulação antes de inverter normais (Corrige erro NotAllTrianglesError)
    bottom = bottom.triangulate()
    bottom = bottom.flip_normals()
    
    # Junta as duas partes
    combined = wall_mesh + bottom
    
    # --- CORREÇÃO DO ERRO DE EXTENSÃO ---
    # O comando extract_surface() garante que o objeto final seja PolyData (superfície),
    # que é o único tipo que o formato .STL aceita.
    return combined.extract_surface().clean().triangulate()

    # Coordenadas da Parede
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    pts = np.column_stack((X.ravel(), Y.ravel(), Zg.ravel()))
    
    # Cria a malha da parede
    wall_grid = pv.StructuredGrid()
    wall_grid.dimensions = [n_theta, n_z, 1]
    wall_grid.points = pts
    
    # Converte a parede para superfície
    wall_mesh = wall_grid.extract_surface().triangulate()
    
    # Cria o fundo
    bottom = pv.Circle(radius=raio_base_real, resolution=n_theta)
    
    # --- CORREÇÃO AQUI ---
    # Primeiro transformamos o círculo em triângulos, 
    # só depois invertemos as normais.
    bottom = bottom.triangulate()
    bottom = bottom.flip_normals()
    
    # Junta as duas partes
    combined = wall_mesh + bottom
    
    # Limpa e finaliza
    return combined.clean().triangulate()

# ==========================================
# 3. INTERFACE GRÁFICA
# ==========================================
st.title("🥤 Criador de Copos Personalizados por função")
st.markdown("Defina as dimensões do copo, visualize-o em 3D e baixe o modelo para impressão.")

col1, col2 = st.columns([1, 2])

#Exibe botões
with col1:
    st.markdown("### ⚙️ Parâmetros")
    r0 = st.number_input("Raio da Base (cm)", 0.5, 20.0, 3.0, step=0.1)
    height = st.number_input("Altura (cm)", 1.0, 50.0, 5.0, step=0.5)
    func_str = st.text_input("f(z)", value="sin(z) + 0.5")
    st.caption("Tente: `z * 0.5` ou `log(z+1)`")
    
    btn_calc = st.button("🔄 Gerar Modelo", key="btn_main")

#Exibe gráfico, volume essas paradas
with col2:
    if btn_calc or func_str:
        f_func = make_f_func(func_str)
        
 
        vol = calcular_volume(r0, height, f_func)
        st.info(f"📊 Volume Estimado: **{(vol*0.001):.2f} litros**")
        
        mesh = gerar_mesh(r0, height, f_func)
        
        if mesh:
            st.caption("Renderizando imagem...")
            try:
 
                plotter = pv.Plotter(off_screen=True, window_size=[600, 400])
                plotter.add_mesh(mesh, color="lightblue", opacity=0.9, show_edges=False, specular=0.5)
                plotter.view_isometric()
                plotter.camera.zoom(1.2)
                
                img_path = "temp_copo.png"
                plotter.screenshot(img_path)
                st.image(img_path, caption="Visualização 3D", use_column_width=True)
                
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    mesh.save(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            label="📥 Baixar Arquivo .STL",
                            data=f,
                            file_name="meu_copo.stl",
                            mime="model/stl"
                        )
            except Exception as e:
                st.error(f"Erro: {e}")
        else:

            st.error("Erro: raio negativo detectado.")




