from pyvista import examples

def test():
    mesh = examples.download_dragon()
    mesh['scalars'] = mesh.points[:, 1]
    mesh.plot(background='white', cpos='xy', cmap='plasma', show_scalar_bar=False)