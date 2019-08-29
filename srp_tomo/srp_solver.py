import numpy as np
import raytracer.refraction as rf
from srp_grid import RandomGrid, WaveBasis
import scipy.sparse.csgraph._shortest_path as sp
import scipy.spatial.qhull as qhull


def interp_weights(xyz, uvw, d=2):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, dim=3):
    if dim == 2:
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    elif dim == 3:
        return np.einsum('inj,nj->in', np.take(values, vtx, axis=1), wts)


def dijkstra_eikonalField_weld(nx, ny, orientations, weld_mask, background, dx,
                               cx, cy, sx, sy, rx, ry, grid=None,
                               mode='orientations'):

    # Define material properties
    rho_parent = 7.9e3
    # Define weld material
    rho_weld = 8.0e3
    c_parent = 1e9*np.array(
        [[255.61, 95.89, 95.89, 0., 0., 0.],
         [95.89, 255.61, 95.89, 0., 0., 0.],
         [95.89, 95.89, 255.61, 0., 0., 0.],
         [0., 0., 0., 79.86, 0., 0.],
         [0., 0., 0., 0., 79.86, 0.],
         [0., 0., 0., 0., 0., 79.86]])
    # This is the stiffness matrix in material coordinates
    c_weld_temp = 1e9*np.array([[240, 118, 148, 0, 0, 0],
                                [118, 240, 148, 0, 0, 0],
                                [148, 148, 220, 0, 0, 0],
                                [0, 0, 0, 100, 0, 0],
                                [0, 0, 0, 0, 100, 0],
                                [0, 0, 0, 0, 0, 61]])

    # c_weld_temp = c_parent
    # rho_weld = rho_parent

    wave_index = 0
    cp_parent, m_parent, cg_parent, _ = rf.calculate_slowness(
        c_parent, rho_parent, 0)
    parent_velo = np.linalg.norm(cg_parent.real, axis=0)[wave_index]
    parent_velo_sq = (parent_velo)**2
    # Reference wave velocity
    cp_weld, m_weld, cg_weld0, _ = rf.calculate_slowness(
        c_weld_temp, rho_weld, 0)

    p_waves = WaveBasis()
    p_waves.set_material_props(c_weld_temp, rho_weld)
    p_waves.calculate_wavespeeds()
    p_waves.define_parent_velo(parent_velo_sq)
    sources = np.column_stack((np.array(sx), np.array(sy)))
    receivers = np.column_stack((np.array(rx), np.array(ry)))
    if grid is None:
#        print('Dijkstra: setting up the grid...')
        nodes_per_subregion = 2
        spacing = dx/nodes_per_subregion

        nx = orientations.shape[1]
        ny = orientations.shape[0]
        grid = RandomGrid(nx, ny, cx, cy, dx, spacing, 0.15,
                          mode=mode)
        grid.generate_realisation(weld_model=orientations, only_weld=False)
        grid.add_mask(weld_mask)
        grid.add_points(sources, receivers)
        grid.get_tree(4)
        grid.set_weld_flag()
    else:
#        print('Dijkstra: loading the grid...')
        grid = grid
        grid.add_mask(weld_mask)
        grid.orientations = orientations
#    print('Dijkstra: calculating graph edges...')
    grid.calculate_edges(p_waves)
#    start = grid.grid_1.shape[0]
#    print('Dijkstra: calculating shortest paths')
    nx = orientations.shape[1]
    ny = orientations.shape[0]
    dd = sp.shortest_path(grid.edges, return_predecessors=False,
                          indices=np.array(range(grid.grid_1.shape[0],
                                                 grid.grid_1.shape[0]
                                                 + len(sx))))
 #   print('Dijkstra: recasting to regular grid...')
    vtx, wts = interp_weights(grid.grid, grid.image_grid)
    dist_structured = interpolate(
        dd, vtx, wts, dim=3).reshape(len(sources), ny, nx).transpose(1, 2, 0)
    dist_structured[0, 0] = dist_structured[[1, 1, 0], [0, 1, 1]].mean(axis=0)
    dist_structured[0, -1] = dist_structured[[1, 1, 0], [-1, -2,
                                                         -2]].mean(axis=0)
    dist_structured[-1, -1] = dist_structured[[-2, -2, -1], [-1, -2,
                                                             -2]].mean(axis=0)
    dist_structured[-1, 0] = dist_structured[[-2, -2, -1], [0, 1,
                                                            1]].mean(axis=0)
    tofs = dd[:, grid.target_idx]

    return grid, dist_structured, tofs.T
