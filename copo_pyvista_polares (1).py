import streamlit as st
import numpy as np
import numexpr as ne
import pyvista as pv
import tempfile
import os

# ==========================================
# CORREÇÃO DA TELA PRETA (HEADLESS)
# ==========================================
# Isso liga um "monitor virtual" para que o servidor consiga desenhar o 3D
try:
    pv.start_xvfb()
except Exception as e:
    # Se der erro aqui, geralmente é porque o xvfb já está rodando ou não foi instalado
    pass

# Configurações globais do PyVista para não tentar abrir janelas pop-up
pv.set_jupyter_backend(None) 
pv.global_theme.background = 'white' # Fundo branco ajuda a ver melhor
pv.global_theme.font.color = 'black'

st.set_page_config(page_title="Design de Copos", layout="wide")

# ==========================================
# FUNÇÕES MATEMÁTICAS
# ==========================================
def make_f_func(expr: str):
    def f(z):
        z_arr = np.asarray(z)
        local_dict = {'z': z_arr, 'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt}
        try:
            return ne.evaluate(expr, local_dict=local_dict)
        except:
            return np.zeros_like(z_arr)
    return f

def calcular_volume(r0, H, f_func, n_z=1000):
    Z = np.linspace(0.0, H, n_z)
    try:
        Rz = r0 + f_func(Z)
        Rz = np.maximum(Rz, 0)
        return np.trapz(np.pi * (Rz**2), Z)
    except:
        return 0.0

def gerar_mesh(r0, H, f_func):
    n_z, n_theta = 150, 100 # Reduzi um pouco a resolução para ser mais leve
    Z = np.linspace(0.0, H, n_z)
    theta = np.linspace(0.0, 2*np.pi, n_theta)
    Tg, Zg = np.meshgrid(theta, Z)
    try:
        Rg = r0 + f_func(Zg)
        Rg = np.maximum(Rg, 0)
    except:
        return None

    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    pts = np.column_stack((X.ravel(), Y.ravel(), Zg.ravel()))
    grid = pv.StructuredGrid()
    grid.dimensions = [n_theta, n_z, 1]
    grid.points = pts
    bottom = pv.Circle(radius=r0, resolution=100)
    return grid.combine(bottom)

# ==========================================
# INTERFACE
# ==========================================
st.title("🥤 Criador de Copos Web")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Parâmetros")
    r0 = st.number_input("Raio da Base (cm)", 0.5, 20.0, 3.0)
    height = st.number_input("Altura (cm)", 1.0, 50.0, 5.0)
    func_str = st.text_input("Função f(z)", value="sin(z) + 0.5")
    btn = st.button("Gerar Visualização 3D")

with col2:
    if btn or func_str:
        f_func = make_f_func(func_str)
        
        # 1. Volume
        vol = calcular_volume(r0, height, f_func)
        st.success(f"Volume do Copo: {vol:.2f} cm³")
        
        # 2. Visualização 3D (Modo Estático / Screenshot)
        mesh = gerar_mesh(r0, height, f_func)
        if mesh:
            with st.spinner("Renderizando 3D..."):
                # Configuração crucial para rodar na nuvem: off_screen=True
                plotter = pv.Plotter(off_screen=True, window_size=[600, 400])
                plotter.add_mesh(mesh, color="lightblue", opacity=0.9, show_edges=False)
                plotter.view_isometric()
                
                # Em vez de mostrar interativo, tiramos uma "foto"
                img = plotter.screenshot(return_img=True)
                st.image(img, caption="Modelo 3D", use_column_width=True)
                
                # Download
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    mesh.save(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 Baixar arquivo .STL", f, "copo.stl")
        else:
            st.error("Erro na geometria. Verifique a função.")
