import streamlit as st
import numpy as np
import numexpr as ne
import pyvista as pv
import tempfile
import os

# ==========================================
# CONFIGURAÇÃO DE SERVIDOR (HEADLESS)
# ==========================================
try:
    pv.start_xvfb() # Inicia o monitor virtual
except:
    pass

# Força o PyVista a não esperar interação humana
pv.OFF_SCREEN = True 
# Define tema para garantir contraste (fundo branco, letras pretas)
pv.global_theme.background = 'white'
pv.global_theme.font.color = 'black'

st.set_page_config(page_title="Design de Copos 3D", layout="wide")

# ==========================================
# FUNÇÕES DO COPO
# ==========================================
def make_f_func(expr: str):
    def f(z):
        z_arr = np.asarray(z)
        local_dict = {'z': z_arr, 'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt, 'pi': np.pi}
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
        area_layers = np.pi * (Rz**2)
        return np.trapz(area_layers, Z)
    except:
        return 0.0

def gerar_mesh(r0, H, f_func):
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
    bottom = pv.Circle(radius=r0, resolution=100)
    return grid.combine(bottom)

# ==========================================
# INTERFACE
# ==========================================
st.title("🥤 Criador de Copos Personalizados")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Parâmetros")
    r0 = st.number_input("Raio da Base (cm)", 0.5, 20.0, 3.0, step=0.1)
    height = st.number_input("Altura (cm)", 1.0, 50.0, 5.0, step=0.5)
    func_str = st.text_input("Curvatura da Parede f(z)", value="sin(z) + 0.5")
    
    # Adicionei um ID único ao botão para evitar recarregamentos falsos
    btn_calc = st.button("🔄 Gerar Copo", key="btn_gerar")

with col2:
    # Sempre tenta rodar se houver input, para não ficar tela branca no início
    if btn_calc or func_str:
        f_func = make_f_func(func_str)
        
        # 1. Volume
        vol = calcular_volume(r0, height, f_func)
        st.info(f"📊 Volume Estimado: **{vol:.2f} cm³**")
        
        # 2. Visualização 3D
        mesh = gerar_mesh(r0, height, f_func)
        
        if mesh:
            st.write("Renderizando modelo...") # Debug visual
            
            # --- BLOCO DE RENDERIZAÇÃO SEGURO ---
            try:
                # Cria o plotter
                plotter = pv.Plotter(off_screen=True, window_size=[600, 400])
                plotter.add_mesh(mesh, color="lightblue", opacity=0.9, show_edges=False, specular=0.5)
                plotter.view_isometric()
                plotter.camera.zoom(1.2)
                
                # TRUQUE: Salva em arquivo físico em vez de memória
                screenshot_path = "copo_temp.png"
                plotter.screenshot(screenshot_path)
                
                # Mostra a imagem lendo do arquivo
                st.image(screenshot_path, caption="Visualização 3D", use_column_width=True)
                
                # Remove o arquivo temporário para não acumular lixo
                # (Opcional, mas boa prática)
                # os.remove(screenshot_path) 
                
                # Botão Download
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                    mesh.save(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 Baixar STL", f, "meu_copo.stl")
                        
            except Exception as e:
                st.error(f"Erro ao renderizar imagem: {e}")
                st.warning("Dica: Verifique se o arquivo 'packages.txt' contém 'xvfb' e 'libgl1-mesa-glx'.")
        else:
            st.error("Erro geométrico: A função gerou um raio negativo ou inválido.")
