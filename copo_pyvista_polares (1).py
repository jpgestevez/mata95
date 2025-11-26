import streamlit as st
import numpy as np
import numexpr as ne
import pyvista as pv
import tempfile
import os

# ==========================================
# 1. CONFIGURAÇÃO DO SERVIDOR (HEADLESS)
# ==========================================
try:
    pv.start_xvfb() # Inicia monitor virtual (Linux/Cloud)
except:
    pass

pv.OFF_SCREEN = True 
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'

st.set_page_config(page_title="Design de Copos 3D", layout="wide")

# ==========================================
# 2. FUNÇÕES MATEMÁTICAS
# ==========================================
def make_f_func(expr: str):
    """Interpreta a função digitada pelo usuário de forma segura."""
    def f(z):
        z_arr = np.asarray(z)
        local_dict = {'z': z_arr, 'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}
        try:
            return ne.evaluate(expr, local_dict=local_dict)
        except:
            return np.zeros_like(z_arr)
    return f

def calcular_volume(r0, H, f_func, n_z=1000):
    """Calcula o volume usando integral numérica."""
    Z = np.linspace(0.0, H, n_z)
    try:
        Rz = r0 + f_func(Z)
        Rz = np.maximum(Rz, 0)
        area_layers = np.pi * (Rz**2)
        return np.trapz(area_layers, Z)
    except:
        return 0.0

def gerar_mesh(r0, H, f_func):
    """Gera a geometria 3D pronta para visualização e STL."""
    n_z, n_theta = 150, 100
    Z = np.linspace(0.0, H, n_z)
    theta = np.linspace(0.0, 2*np.pi, n_theta)
    Tg, Zg = np.meshgrid(theta, Z)
    
    try:
        Rg = r0 + f_func(Zg)
        Rg = np.maximum(Rg, 0)
    except:
        return None

    # Coordenadas da parede
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    pts = np.column_stack((X.ravel(), Y.ravel(), Zg.ravel()))
    
    # Cria a malha da parede
    wall = pv.StructuredGrid()
    wall.dimensions = [n_theta, n_z, 1]
    wall.points = pts
    
    # Cria a malha do fundo
    bottom = pv.Circle(radius=r0, resolution=100)
    
    # Junta as duas partes
    combined = wall + bottom
    
    # --- A CORREÇÃO MÁGICA ---
    # Converte o objeto complexo em uma superfície simples de triângulos.
    # Isso permite salvar como .STL sem dar erro de extensão.
    return combined.extract_surface().triangulate()

# ==========================================
# 3. INTERFACE GRÁFICA
# ==========================================
st.title("🥤 Criador de Copos Personalizados")
st.markdown("Defina a geometria do copo, visualize em 3D e baixe o modelo para impressão.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Parâmetros")
    r0 = st.number_input("Raio da Base (cm)", 0.5, 20.0, 3.0, step=0.1)
    height = st.number_input("Altura (cm)", 1.0, 50.0, 5.0, step=0.5)
    func_str = st.text_input("Curvatura f(z)", value="sin(z) + 0.5")
    st.caption("Tente: `z * 0.5` ou `log(z+1)`")
    
    # Botão com ID único
    btn_calc = st.button("🔄 Gerar Modelo", key="btn_main")

with col2:
    if btn_calc or func_str:
        f_func = make_f_func(func_str)
        
        # Exibe Volume
        vol = calcular_volume(r0, height, f_func)
        st.info(f"📊 Volume Estimado: **{vol:.2f} cm³**")
        
        # Gera e Exibe 3D
        mesh = gerar_mesh(r0, height, f_func)
        
        if mesh:
            st.caption("Renderizando imagem...")
            try:
                # Plota off-screen
                plotter = pv.Plotter(off_screen=True, window_size=[600, 400])
                plotter.add_mesh(mesh, color="lightblue", opacity=0.9, show_edges=False, specular=0.5)
                plotter.view_isometric()
                plotter.camera.zoom(1.2)
                
                # Salva imagem temporária
                img_path = "temp_copo.png"
                plotter.screenshot(img_path)
                st.image(img_path, caption="Visualização 3D", use_column_width=True)
                
                # Botão Download STL
                # Agora deve funcionar porque usamos .extract_surface().triangulate()
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
                st.error(f"Erro técnico: {e}")
        else:
            st.error("Erro na geometria: raio negativo detectado.")
