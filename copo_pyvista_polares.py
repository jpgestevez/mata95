#!/usr/bin/env python3
\"\"\"Copo personalizado — integração dupla (polos) + visualização PyVista
Uso:
    python copo_pyvista_polares.py --r0 3 --height 2 --func "z**3" --show --save_mesh copo.stl

Dependências:
    numpy, numexpr, pyvista, vtk, argparse
\"\"\"
import argparse
import numpy as np
import numexpr as ne
import pyvista as pv
import os

# -----------------------------
# Helpers
# -----------------------------
def make_f_func(expr: str):
    \"\"\"Retorna função f(z) que avalia expressão usando numexpr (seguro para arrays).
    A expressão deve usar 'z' como variável.
    \"\"\"
    # Teste simples para garantir que é avaliada
    def f(z):
        z_arr = np.asarray(z)
        try:
            return ne.evaluate(expr, local_dict={'z': z_arr, 'np': np})
        except Exception as e:
            raise ValueError(f\"Erro ao avaliar expressão f(z): {e}\")
    # testar com escalar
    try:
        _ = float(f(0.0))
    except Exception as e:
        raise ValueError(f\"Expressão inválida para f(z): {e}\")
    return f

# -----------------------------
# Volume - integração dupla em coordenadas polares (numérica por camadas em z)
# Para cada z, V_layer = 
#   ∫_{theta=0}^{2pi} ∫_{rho=0}^{r(z)} rho d rho d theta = pi * r(z)^2
# Mas implementamos numericamente por camada para deixar explícito o cálculo duplo.
# -----------------------------

def volume_polar_numeric(r0, H, f_func, n_z=1000):
    """Calcula o volume aproximado usando integração numérica em coordenadas polares.
    Implementação eficiente: para cada z, calculamos area do disco pi*r(z)^2 e somamos.
    """
    Z = np.linspace(0.0, H, n_z)
    Rz = r0 + f_func(Z)
    area_layers = np.pi * (Rz**2)
    volume = np.trapz(area_layers, Z)
    return volume

# -----------------------------
# Geração de mesh via rotação do perfil (pyvista StructuredGrid)
# -----------------------------
def gerar_mesh_pyvista(r0, H, f_func, n_z=200, n_theta=120):
    Z = np.linspace(0.0, H, n_z)
    theta = np.linspace(0.0, 2*np.pi, n_theta)
    Tg, Zg = np.meshgrid(theta, Z)
    Rg = r0 + f_func(Zg)
    X = Rg * np.cos(Tg)
    Y = Rg * np.sin(Tg)
    pts = np.column_stack((X.ravel(), Y.ravel(), Zg.ravel()))
    grid = pv.StructuredGrid()
    grid.dimensions = [n_theta, n_z, 1]
    grid.points = pts
    return grid

# -----------------------------
# Salvar o fundo (disco) e anexar ao mesh
# -----------------------------
def criar_fundo_pyvista(r0, n_points=200):
    theta = np.linspace(0.0, 2*np.pi, n_points)
    xb = r0 * np.cos(theta)
    yb = r0 * np.sin(theta)
    zb = np.zeros_like(theta)
    poly = pv.PolyData(np.column_stack([xb, yb, zb]))
    disk = poly.delaunay_2d()  # triangula o interior do polígono circular
    return disk

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Copo personalizado — integração polar + PyVista')
    parser.add_argument('--r0', type=float, default=3.0, help='Raio do fundo (r0)')
    parser.add_argument('--height', type=float, default=2.0, help='Altura do copo (H)')
    parser.add_argument('--func', type=str, default='z**3', help='Função f(z) que define a parede')
    parser.add_argument('--n_z', type=int, default=1000, help='Número de camadas em z para integração')
    parser.add_argument('--show', action='store_true', help='Mostrar visualização PyVista')
    parser.add_argument('--save_mesh', type=str, default=None, help='Salvar mesh final (ex: copo.stl ou copo.obj)')
    args = parser.parse_args()

    # validar
    if args.height <= 0:
        raise SystemExit('Altura deve ser positiva')
    if args.r0 < 0:
        raise SystemExit('Raio do fundo deve ser não-negativo')

    f_func = make_f_func(args.func)

    # verificar r(z) não-negativo em [0,H]
    Ztest = np.linspace(0.0, args.height, 50)
    Rt = args.r0 + f_func(Ztest)
    if np.any(Rt < 0):
        raise SystemExit('Raio r(z) ficou negativo em algum ponto do intervalo. Ajuste f(z) ou r0.')

    vol = volume_polar_numeric(args.r0, args.height, f_func, n_z=args.n_z)
    print(f'Volume (integração polar numérica) = {vol:.8f}')

    # gerar mesh
    mesh = gerar_mesh_pyvista(args.r0, args.height, f_func, n_z=300, n_theta=240)
    bottom = criar_fundo_pyvista(args.r0, n_points=256)

    # combinar meshes: vamos unir o mesh das paredes com o disco do fundo
    combined = mesh.combine(bottom)

    # salvar se solicitado
    if args.save_mesh:
        out = args.save_mesh
        try:
            combined.save(out)
            print(f'Mesh salvo em {out}')
        except Exception as e:
            print('Erro ao salvar mesh:', e)

    if args.show:
        p = pv.Plotter()
        p.add_mesh(combined, color='lightblue', opacity=1.0)
        p.add_axes()
        p.show()

if __name__ == '__main__':
    main()
