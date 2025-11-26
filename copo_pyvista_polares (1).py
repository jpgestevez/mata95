import streamlit as st
import numpy as np
import numexpr as ne
import pyvista as pv
import tempfile
import os

# ==========================================
# CONFIGURAÇÃO HEADLESS (ESSENCIAL PARA NUVEM)
# ==========================================
# Inicia o monitor virtual (Xvfb) para que o PyVista consiga renderizar sem tela física
# Isso requer que o 'xvfb' esteja instalado via packages.txt
try:
    pv.start_xvfb()
except Exception:
    pass

# Força o modo off-screen globalmente para evitar erros de janela
pv.OFF_SCREEN = True

st.set_page_config(page_title="Design de Copos 3D", layout="wide")

# ==========================================
# LÓGICA DO COPO
# ==========================================
def make_f_func(expr: str):
    """Converte a string do usuário em uma função matemática segura."""
    def f(z):
        z_arr = np.asarray(z)
        local_dict = {'z': z_arr, 'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}
        try:
            return ne.evaluate(expr, local_dict=local_dict)
        except:
            return np.zeros_like(z_arr)
    return f

def calcular_volume(r0, H, f_func, n_z=1000):
    """Integração numérica do volume."""
    Z = np.linspace(0.0, H, n_z)
    try:
        Rz = r0 + f_func(Z)
        Rz = np.maximum(Rz, 0) # Raio não pode ser negativo
        area_layers = np.pi * (Rz**2)
        return np.trapz(area_layers, Z)
    except:
        return 0.0

def gerar_mesh(r0, H, f_func):
    """Gera a malha 3D do copo."""
    n_z, n_theta = 150, 100
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
    
    # Fundo do copo
    bottom = pv.Circle(radius=r0, resolution=100)
    return grid.combine(bottom)

# ==========================================
# INTERFACE GRÁFICA
# ==========================================
st.title("🥤 Criador de Copos Personalizados")
st.markdown("Altere os parâmetros abaixo para projetar seu copo e calcular o volume.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Parâmetros")
    r0 = st.number_input("Raio da Base (cm)", 0.5, 20.0, 3.0, step=0.1)
    height = st.number_input("Altura (cm)", 1.0, 50.0, 5.0, step=0.5)
    func_str = st.text_input("Curvatura da Parede f(z)", value="sin(z) + 0.5")
    st.caption("Exemplos: `z * 0.5`, `sin(z)`, `log(z+1)`")
    
    btn_calc = st.button("🔄 Gerar Copo", type="primary")

with col2:
    if btn_calc or func_str:
        f_func = make_f_func(func_str)
        
        # 1. Cálculo e Exibição do Volume
        vol = calcular_volume(r0, height, f_func)
        st.info(f"📊 Volume Estimado: **{vol:.2f} cm³** (ml)")
        
        # 2. Visualização 3D
        mesh = gerar_mesh(r0, height, f_func)
        
        if mesh:
            # Renderização Estática (Mais segura para Web)
            with st.spinner("Renderizando modelo 3D..."):
                try:
                    # Configuração do Plotter
                    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
                    plotter.add_mesh(mesh, color="lightblue", opacity=0.9, show_edges=False, specular=0.5)
                    plotter.view_isometric()
                    plotter.camera.zoom(1.2)
                    
                    # Gera imagem
                    img = plotter.screenshot(return_img=True)
                    st.image(img, caption="Modelo 3D do Copo", use_column_width=True)
                    
                    # Botão de Download
                    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                        mesh.save(tmp.name)
                        with open(tmp.name, "rb") as f:
                            st.download_button(
                                label="📥 Baixar Arquivo .STL (Impressão 3D)",
                                data=f,
                                file_name="meu_copo.stl",
                                mime="model/stl"
                            )
                except Exception as e:
                    st.error(f"Erro na renderização 3D: {e}")
                    st.warning("Verifique se o arquivo packages.txt contém 'libgl1-mesa-glx' e 'xvfb'.")
        else:
            st.error("Não foi possível gerar a geometria. Verifique se a função matemática é válida.")
