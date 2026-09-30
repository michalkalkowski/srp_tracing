"""
Part of srp_tracing.
Test of the SRP ray tracing routine for a triple-V (three-level, staircase
bevel) austenitic weld, using the backwall-relay pulse-echo configuration
(see test_ogilvy_backwall_relay.py) on the simplified, trimmed grid (see
test_ogilvy_simpl.py) rather than the staircased RectGrid.

Geometry: three stacked V-levels with independent bevel angles, from a
least-squares fit of the real A6 weld's cross-section (the boundary of its
EBSD grain-orientation map, ../../srp_tft/data/A6_axis_orientations_1mm.npy)
to gen_ogilvy.multi_weld.MultiVWeld -- level heights 7.645 / 16.268 / 11.087
mm with bevel half-angles 38.8 / 24.6 / 13.1 deg and a 1.32 mm root offset
(see ../../srp_tft/examples/_a6.py, A6_LEVELS/A6_ROOT). Orientations are
the Ogilvy model evaluated on that fitted geometry (zeta=0, auto-fitted T,
n=1 -- the same starting point ../../srp_tft/examples/a6_matrix.py's blind
fit uses), not the real measured EBSD grains: this is an illustrative
"Ogilvy weld with a triple-V chamfer", not an EBSD-validated one.

Sources and receivers sit centrally on the top surface; the signal reflects
off a flat backwall at the root level, combined via
solver.combine_via_boundary() rather than a mirrored/doubled domain.

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from srp_tracing import grid, solver
from gen_ogilvy.multi_weld import MultiVWeld, WeldLevel

# Triple-V weld geometry: three stacked V-levels (height, bevel half-angle
# deg), bottom to top, fitted to the real A6 weld outline -- see module
# docstring.
THICKNESS = 35.0
A6_ROOT = 1.32
A6_LEVELS = ((7.645, 38.784), (16.268, 24.577), (35.0 - 7.645 - 16.268, 13.056))
dx = 1.0

weld = MultiVWeld(thickness=THICKNESS, d_L=A6_ROOT, d_R=A6_ROOT,
                  levels=[WeldLevel(height=h, alpha_L=alpha, alpha_R=alpha)
                          for h, alpha in A6_LEVELS])
weld.calculate_weld_cross_section()
weld.define_grid_size(dx, use_centroids=True, add_boundary_cells=True,
                      boundary_offset=0.4)
weld.assign_weld_params(dict(zeta=0., T_L=1., T_R=1., n_L=1., n_R=1.))
weld.calculate_grain_orientations(auto_T=True)

# Define the domain -- pad with parent metal on both sides of the weld
# (85 total columns, matching ../../srp_tft/examples/_a6.py's own padded
# plate width)
ny, nx_weld = weld.in_weld.shape
pad = (85 - nx_weld)//2

def embed(values):
    return np.column_stack((np.zeros((ny, pad)), values, np.zeros((ny, pad))))

orientation_map = embed(weld.theta_full)
new_wm = embed(weld.in_weld)
nx = orientation_map.shape[1]

# The real outline (left chamfer bottom-to-top, reversed, then right
# chamfer bottom-to-top): x increasing throughout with the flat root in
# the middle, the ordering srp_tracing's trim_to_weld requires (see
# ../../srp_tft/examples/_a6.py's geometry() for the same construction on
# the real EBSD mask).
outline = np.concatenate((weld.left_chamfer[::-1], weld.right_chamfer), axis=0)

# Sensors, on the top surface
start_gen = -32.55
pitch = 2.05
element_width = 1.55

sx = (element_width/2 + np.arange(start_gen, start_gen + 32*pitch, pitch))
sources = np.column_stack((sx, np.full(len(sx), THICKNESS)))

# A grid of points along the flat backwall at y=0, independent of no_seeds
backwall_x = np.arange(-40, 40, 0.5)
backwall = np.column_stack((backwall_x, np.zeros_like(backwall_x)))

# Properties: isotropic parent, transversely isotropic weld -- the same
# nominal mock-up materials as ../../srp_tft/examples/_iweld.py's
# materials()
rho = 8.0
c_long_parent = 5.84
poisson_parent = 0.33
c_shear_parent = c_long_parent*np.sqrt((1 - 2*poisson_parent)/(2*(1 - poisson_parent)))
mu = rho*c_shear_parent**2
lam = rho*c_long_parent**2 - 2*mu
c_parent = np.diag([lam + 2*mu]*3 + [mu]*3)
c_parent[:3, :3] += lam*(1 - np.eye(3))

c11, c12, c23, c33, c44 = 249., 91., 133., 207., 117.
c_weld = np.array([[c11, c12, c23, 0, 0, 0],
                   [c12, c11, c23, 0, 0, 0],
                   [c23, c23, c33, 0, 0, 0],
                   [0, 0, 0, c44, 0, 0],
                   [0, 0, 0, 0, c44, 0],
                   [0, 0, 0, 0, 0, (c11 - c12)/2]])

parent_basis = grid.WaveBasis(anisotropy=0, velocity_variant='group')
parent_basis.set_material_props(c_parent, rho)
parent_basis.calculate_wavespeeds()

weld_basis = grid.WaveBasis(anisotropy=1, velocity_variant='group')
weld_basis.set_material_props(c_weld, rho)
weld_basis.calculate_wavespeeds(angles_from_ray=True)

cx = 0
cy = ny*dx/2  # domain spans y in [0, ny*dx]: y=0 is the real backwall

no_seeds = 8

test_grid = grid.SimplRectGrid(nx, ny, cx, cy, dx, no_seeds)
test_grid.assign_model(mode='orientations', property_map=orientation_map)
test_grid.assign_materials(new_wm, dict([(0, parent_basis), (1, weld_basis)]))
test_grid.trim_to_weld(outline, mirror_domain=False)
test_grid.simplify_grid(left_add=pad, right_add=pad)
test_grid.add_points(points=np.r_[sources, backwall],
                     sources=np.arange(len(sources)),
                     targets=np.arange(len(sources), len(sources) + len(backwall)))
# set_up_graph + update_edges (not calculate_graph): on a weld this narrow
# at the root (2*A6_ROOT = 2.64 mm), calculate_graph adds isotropic chords
# straight through it that shorten some times by up to a few hundredths of
# a us -- see ../../srp_tft/examples/_a6.py's forward(), which hits the
# same issue on the same geometry.
test_grid.set_up_graph()
test_grid.update_edges()

test = solver.Solver(test_grid)
test.solve(source_indices=test_grid.source_idx, with_points=True)

times_to_backwall = test.tfs[:, test_grid.target_idx]
tofs_pulse_echo, via_index = solver.combine_via_boundary(times_to_backwall)

print('Pulse-echo times (us), 3 sensor pairs:')
print(tofs_pulse_echo[[5, 15, 26]][:, [5, 15, 26]])

plt.figure()
plt.plot(tofs_pulse_echo[10], lw=1, c='red', label='sensor 10 -> all')
plt.plot(tofs_pulse_echo[16], lw=1, c='C0', label='sensor 16 -> all')
plt.xlabel('sensor #')
plt.ylabel('pulse-echo time of flight in us')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting a reflected ray path
source_node = test_grid.source_idx[10]
target_node = test_grid.source_idx[16]
via_node = test_grid.target_idx[via_index[10, 16]]
path = test.calculate_relay_ray_path(source_node, target_node, via_node)

fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(12, 6),
                                  gridspec_kw={'width_ratios': [2, 1]})
ax.imshow(np.rad2deg(orientation_map), origin='lower',
          extent=[cx - nx*dx/2, cx + nx*dx/2, 0, ny*dx], cmap='twilight')
ax.plot(test_grid.grid[:, 0], test_grid.grid[:, 1], 'o', ms=1, c='gray')
ax.plot(outline[:, 0], outline[:, 1], '-', lw=1, c='white')
ax.plot(backwall[:, 0], backwall[:, 1], '.', ms=2, c='cyan')
ax.plot(test_grid.grid[path, 0], test_grid.grid[path, 1], '-x', ms=4, c='yellow')
zoom_box = (-16, -6, 5, 15)
ax.add_patch(plt.Rectangle((zoom_box[0], zoom_box[2]),
                           zoom_box[1] - zoom_box[0], zoom_box[3] - zoom_box[2],
                           fill=False, ec='red', lw=1))
ax.set_xlim(-45, 45)
ax.set_aspect('equal')

# Zoomed-in panel: at the full-domain scale and ms=1 markers above, the
# chamfer seeds (now node_spacing = pixel_size/no_seeds = 0.125 mm apart,
# see trim_to_weld) are sub-pixel and indistinguishable from the old,
# 4x-sparser default -- zoom in on the outlined region to actually see the
# per-seed density on one bevel transition.
on_chamfer = ((test_grid.grid[:, 0] >= zoom_box[0]) & (test_grid.grid[:, 0] <= zoom_box[1])
             & (test_grid.grid[:, 1] >= zoom_box[2]) & (test_grid.grid[:, 1] <= zoom_box[3]))
ax_zoom.plot(test_grid.grid[on_chamfer, 0], test_grid.grid[on_chamfer, 1],
            'o', ms=4, c='red')
ax_zoom.plot(outline[:, 0], outline[:, 1], '-', lw=0.5, c='gray')
ax_zoom.set_xlim(zoom_box[0], zoom_box[1])
ax_zoom.set_ylim(zoom_box[2], zoom_box[3])
ax_zoom.set_title(f'zoom: {on_chamfer.sum()} nodes in the red box')
ax_zoom.set_aspect('equal')

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, 'ogilvy_triple_v_ray_path.png'), dpi=150)

plt.show()
