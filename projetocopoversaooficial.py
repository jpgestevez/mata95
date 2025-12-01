def gerar_mesh(r0, H, f_func):
    """Gera a visualização com fundo conectado corretamente."""
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
    
    # --- CORREÇÃO AQUI ---
    # Primeiro transformamos o círculo em triângulos, 
    # só depois invertemos as normais.
    bottom = bottom.triangulate()
    bottom = bottom.flip_normals()
    
    # Junta as duas partes
    combined = wall_mesh + bottom
    
    # Limpa e finaliza
    return combined.clean().triangulate()
