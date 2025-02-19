import numpy as np
import matplotlib.pyplot as plt

def load_charges(filename):
    data = np.loadtxt(filename)
    return data

def compute_flux_box(charges, x_bounds, y_bounds, z_bounds, density):
    epsilon_0 = 8.854187817e-12
    total_flux = 0
    
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds
    
    dx = (x_max - x_min) / (density)
    dy = (y_max - y_min) / (density)
    dz = (z_max - z_min) / (density)
    
    def electric_field_at_point(point):
        E_total = np.array([0.0, 0.0, 0.0])
        for charge in charges:
            Q, x_c, y_c, z_c = charge
            r_vec = np.array(point) - np.array([x_c, y_c, z_c])
            r = np.linalg.norm(r_vec)
            if r > 0:
                E = (Q / (4 * np.pi * epsilon_0 * r**3)) * r_vec
                E_total += E
        return E_total
    
    def integrate_surface(a_range, b_range, c_fixed, normal, da, db):
        flux = 0
        a_vals = np.linspace(a_range[0], a_range[1], density)
        b_vals = np.linspace(b_range[0], b_range[1], density)
        dA = da * db
        for a in a_vals:
            for b in b_vals:
                point = [a, b, c_fixed]
                E = electric_field_at_point(point)
                flux += np.dot(E, normal) * dA
        return flux * 6
    
    total_flux += integrate_surface((x_min, x_max), (y_min, y_max), z_min, [0, 0, -1], dx, dy)
    total_flux += integrate_surface((x_min, x_max), (y_min, y_max), z_max, [0, 0, 1], dx, dy)
    total_flux += integrate_surface((y_min, y_max), (z_min, z_max), x_min, [-1, 0, 0], dy, dz)
    total_flux += integrate_surface((y_min, y_max), (z_min, z_max), x_max, [1, 0, 0], dy, dz)
    total_flux += integrate_surface((x_min, x_max), (z_min, z_max), y_min, [0, -1, 0], dx, dz)
    total_flux += integrate_surface((x_min, x_max), (z_min, z_max), y_max, [0, 1, 0], dx, dz)
    
    return total_flux

def compute_flux_sphere(charges, sphere_center, radius, grid_density):
    epsilon_0 = 8.854187817e-12
    total_flux = 0
    
    phi_values = np.linspace(0, 2 * np.pi, grid_density)
    theta_values = np.linspace(0, np.pi, grid_density)
    
    for theta in theta_values:
        for phi in phi_values:
            x = sphere_center[0] + radius * np.sin(theta) * np.cos(phi)
            y = sphere_center[1] + radius * np.sin(theta) * np.sin(phi)
            z = sphere_center[2] + radius * np.cos(theta)
            
            E_total = np.array([0.0, 0.0, 0.0])
            for charge in charges:
                Q, x_c, y_c, z_c = charge
                r_vec = np.array([x - x_c, y - y_c, z - z_c])
                r = np.linalg.norm(r_vec)
                if r > 0:
                    E = (Q / (4 * np.pi * epsilon_0 * r**3)) * r_vec
                    E_total += E
            
            dA = (radius**2) * np.sin(theta) * (np.pi / grid_density) * (2 * np.pi / grid_density)
            total_flux += np.dot(E_total, np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])) * dA
    
    return total_flux

def plot_error_vs_density(charges, x_bounds, y_bounds, z_bounds, theoretical_flux):
    densities = np.arange(10, 100, 10)
    errors = []
    
    for density in densities:
        calculated_flux = compute_flux_box(charges, x_bounds, y_bounds, z_bounds, density)
        error = abs(theoretical_flux - calculated_flux)
        errors.append(error)
    
    plt.plot(densities, errors, marker='o')
    plt.xlabel("Grid Density")
    plt.ylabel("Error")
    plt.title("Error vs. Grid Density")
    plt.grid()
    plt.show()

def main():
    filename = "chargesN.dat"
    charges = load_charges(filename)

    x_bounds = (-3, 3)
    y_bounds = (-1, 4)
    z_bounds = (-9, 1)
    grid_density = 50
    
    flux_box = compute_flux_box(charges, x_bounds, y_bounds, z_bounds, grid_density)
    print(f"Flux through box: {flux_box:.5e}")
    
    theoretical_flux = np.sum(charges[:, 0]) / (8.854187817e-12)
    print(f"Theoretical flux: {theoretical_flux:.5e}")
    
    plot_error_vs_density(charges, x_bounds, y_bounds, z_bounds, theoretical_flux)
    
    sphere_center = (0, 0, 0)
    radius = 5
    flux_sphere = compute_flux_sphere(charges, sphere_center, radius, grid_density)
    print(f"Flux through sphere: {flux_sphere:.5e}")
    
if __name__ == "__main__":
    main()
