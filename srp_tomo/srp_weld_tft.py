import numpy as np
#import matlab.engine
import raytracer.refraction as rf
from srp_solver import dijkstra_eikonalField_weld
import scipy.spatial.qhull as qhull
from ray_project_orientations import ray_project


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


def calculate_sec_der(field, dx):
    d2fdx2 = np.zeros(field.shape)
    d2fdy2 = np.zeros(field.shape)
    d2fdxdy = np.zeros(field.shape)

    d2fdx2[1:-1, :] = (field[:-2] + -2*field[1:-1] + field[2:])/(dx**2)
    d2fdy2[:, 1:-1] = (field[:, :-2] + -2*field[:, 1:-1]
                       + field[:, 2:])/(dx**2)

    d2fdxdy[1:-1, 1:-1] = 2./4*1./(dx**2)*(-field[:-2, :-2] + -field[2:, 2:] +
                                           field[:-2, 2:] + field[2:, :-2])
    return d2fdx2, d2fdy2, d2fdxdy


def calculate_first_der(field, dx):
    dfdx = np.zeros(field.shape)
    dfdy = np.zeros(field.shape)

    dfdx[:-1, :] = 1/dx*(field[1:] - field[:-1])
    dfdy[:, :-1] = 1/dx*(field[:, 1:] - field[:, :-1])

    return dfdx, dfdy


def get_grad_1st(field, dx):
    dfdx, dfdy = calculate_first_der(field, dx)
    cost_fn = np.sum(dfdx[:-1]**2) + np.sum(dfdy[:, :-1]**2)
    grad = np.zeros(field.shape)

    for i in range(1, field.shape[0]):
        for j in range(1, field.shape[1]):
            grad[i, j] = -(2*dfdx[i - 1, j]*(-1)/dx + 2*dfdx[i, j]*1/dx
                           + 2*dfdy[i, j - 1]*(-1)/dx + 2*dfdy[i, j]/dx)

    return cost_fn, grad


def get_grad_2nd(field, dx):
    d2fdx2, d2fdy2, d2fdxdy = calculate_sec_der(field, dx)

    cost_fn = np.sum(d2fdx2[1:-1, 1:-1]**2 + d2fdy2[1:-1, 1:-1]**2
                     + d2fdxdy[1:-1, 1:-1]**2)
    grad = np.zeros(field.shape)

    for i in range(1, field.shape[0] - 1):
        for j in range(1, field.shape[1] - 1):
            grad[i, j] = 1/dx**2*(
                2*(d2fdx2[i - 1, j] + -2*d2fdx2[i, j] + d2fdx2[i + 1, j]) +
                2*(d2fdy2[i, j - 1] + -2*d2fdy2[i, j] + d2fdy2[i, j - 1]) +
                4*(d2fdxdy[i - 1, j - 1]*(-1) + d2fdxdy[i + 1, j + 1]*(-1) +
                   d2fdxdy[i - 1, j + 1] + d2fdxdy[i + 1, j - 1]))
    return cost_fn, grad


def get_grad_tot_var(field):
#     field_0 = field[0, 0]
#     isBg = (field == field_0)
    g = np.zeros(field.shape)
    g[1:, 1:] = ((field[1:, 1:] - field[:-1, 1:])**2
                 + (field[1:, 1:] - field[1:, :-1])**2)

    cost_fn = np.sum(g**0.5)

    grad = np.zeros(field.shape)
    for i in range(1, field.shape[0] - 1):
        for j in range(1, field.shape[1] - 1):
            if g[i, j] != 0:
                grad[i, j] = (grad[i, j] + g[i, j]**(-0.5)
                              * (2*field[i, j] - field[i - 1, j]
                                 - field[i, j - 1]))
            if g[i + 1, j] != 0:
                grad[i, j] = (grad[i, j] + g[i + 1, j]**(-0.5)
                              * -(field[i + 1, j] - field[i, j]))
            if g[i, j + 1] != 0:
                grad[i, j] = (grad[i, j] + g[i, j + 1]**(-0.5)
                              * -(field[i, j + 1] - field[i, j]))
    return cost_fn, grad



def timeOfFlight(tMat, c0, sx, sy, rx, ry, gx, gy, nIts, gradFact, epsTV, eps,
                 eps2, bgC=None, bgGx=None, bgGy=None, weld_mask=None):
    """
    timeOfFlight - performs time of flight tomography imaging algorithm.
    
    Parameters:
    ---
    tMat: ndarray, matrix of (relative) arrival times for each send receive 
           pair; first dim receiver, second source
           NOTE: NaN values are ignored. This is convenient if you have
           certain signals that you don't want the algorithm to attempt to
           fit.
    c0: float, background sound speed
    sx: ndarray, source x-positions
    sy: ndarray, source y-positions
    rx: ndarray, receiver x-positions
    ry: ndarray, receiver y-positions
    gx: ndarray, 1-D array of x grid line locations (evenly spaced)
    gy: ndarray, 1-D array of y grid line locations (evenly spaced)
    nIts: int, number of iterations
    gradFact: float, fudge factor to multiply by the gradient; Use to adjust 
                rate of convergence.
    epsTV: float, weightning for TV regularisation
    eps: float, weightning for first derivative regularisation
    eps2:float, weightning for second derivative regulatisation; set to 0 if unwanted.
    bgC: ndarray, starting background values, optional - currently disabled in
            the Python version
    bgGx: ndarray, starting x-grid lines, optional - currently disabled in
            the Python version
    bgGy: ndarray, starting y-grid lines, optional - currently disabled in
            the Python version


    Returns:
    ---
    tftVel: ndarray, image produced, x first dimension, y second dimension

    Written by P. Huthwaite, June 2011
    Regularisation added September 2011 (PH)

    Moved to Python by M Kalkowski Jun 2018
    """

    # Start matlab engine
    #eng = matlab.engine.start_matlab()

    #sx_m = matlab.double(sx.tolist())
    #sy_m = matlab.double(sy.tolist())
    #rx_m = matlab.double(rx.tolist())
    #ry_m = matlab.double(ry.tolist())

    dist_counter = 0
    npx = len(gx)
    npy = len(gy)
    dpx = gx[2] - gx[1]
    cpx = gx.mean()
    cpy = gy.mean()

    dpx = float(dpx)
    cpx = float(cpx)
    cpy = float(cpy)
    # Maybe this could be optimised for performance, but this value is fine
    # for now. It's the distance along the ray that we step each time.
    ds = dpx*.5

    # Set up incident (background) field
#     tftVel = np.ones([npx, npy])*c0
#     s = 1/tftVel
    s = np.zeros([npx - 1, npy - 1])
    # get distortion for inc field
    dijk_grid, dist, tMatInc = dijkstra_eikonalField_weld(npx, npy, s.T,
                                                          weld_mask, 1/c0, dpx,
                                                          cpx, cpy, sx, sy, rx,
                                                          ry,
                                                          mode='orientations')
    dist = dist.transpose(2, 0, 1)
    #dist_m = eng.eikonalField(matlab.double(s.tolist()), 1./c0, dpx, cpx,
    #                          cpy, sx_m, sy_m)
    #dist = np.asarray(dist_m)

    # Set up slowness derivatives
    angles = np.linspace(0, 2*np.pi, 200)
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
    cp, m, cg, _ = rf.calculate_slowness(c_weld_temp, rho_weld, angles)
    sgp = 1/np.linalg.norm(cg.real, axis=1)[:, 0]
    d_angle = angles[1] - angles[0]
    # add points at the start and end of sgp
    before = np.array([-4*d_angle, -3*d_angle, -2*d_angle, -d_angle])
    after = angles[-1] + np.array([d_angle, 2*d_angle, 3*d_angle, 4*d_angle])
    sgp_before = sgp[[-5, -4, -3, -2]]
    sgp_after = sgp[[1, 2, 3, 4]]
    angles = np.concatenate((before, angles, after))
    sgp = np.concatenate((sgp_before, sgp, sgp_after))

    dist_counter = dist_counter + 1;
    nSrc = len(sx)
    nRec = len(rx)
    Gx, Gy = np.meshgrid(gx, gy)
    grid = np.c_[Gx.flatten(), Gy.flatten()]
    receivers = np.column_stack((rx, ry))
    # Interpolate arrival times at transducer locotions from field 
    # vtx, wts = interp_weights(grid, receivers)
    # tMatInc = interpolate(dist.transpose(2, 0, 1).reshape(nSrc, -1), vtx, wts,
    #                       dim=3).T
    # tMatInc = np.zeros(tMat.shape)
    # for cnt in range(nSrc):
    #     tMatInc[:, cnt] = griddata(grid, dist[:, :, cnt].flatten(), receivers)
   
    # First guess is that the background is homogeneous, so first set of 
    # tt estimates (relative to incident field) is zero.
    # Therefore, the first set of residuals is the given tt values.
    resMat = tMat;
    # Calculate derivatives of the slowness field
    R, gradR = get_grad_2nd(s, dpx)
    F, gradF = get_grad_1st(s, dpx)

    resMat[~np.isfinite(resMat)] = 0
        
    costFn = np.sum(np.sum(resMat**2)) + R*eps2 + F*eps
    
    grad = ray_project(dpx, cpx, cpy, sx, sy, rx, ry,
                       np.ascontiguousarray(dist), ds, -resMat, angles, sgp,
                       np.ascontiguousarray(s))
    grad = weld_mask.T*(np.asarray(grad.T) + gradR*eps2 + gradF*eps)
    
    descDir = np.zeros(s.shape)

    # backward line tracing - constants
    beta = 0.3
    alpha = 0.07

    print('Cost function: {}'.format(costFn))
    
    #   imFig = figure;
    # s0 = 1./c0

    for itCnt in range(nIts):
        print('Iteration {}:'.format(itCnt))
        if itCnt > 0:
            denom = np.sum(prev_grad**2)
            numer = np.sum(grad**2) - np.sum((grad.T).dot(prev_grad))
            gamma = numer/denom
            prev_grad = grad
        else:
            gamma = 0
            prev_grad = grad

        descDir = -grad*gradFact + gamma*descDir

        mu = np.sum(np.sum(descDir*grad))

        t = 1
        lineSearchCnt = 0
        while 1:
            lineSearchCnt = lineSearchCnt + 1
            sTest = s + descDir*t

            # figure(imFig)
            # cartimagesc(1./sTest)
            # axis image
            # colorbar
            # drawnow
            _, dist, tMatTest = dijkstra_eikonalField_weld(npx, npy, sTest.T,
                                                          weld_mask, 1/c0, dpx,
                                                          cpx, cpy, sx, sy, rx,
                                                          ry,
                                                           mode='orientations',
                                                          grid=dijk_grid)
            dist = dist.transpose(2, 0, 1)
            #dist_m = eng.eikonalField(matlab.double(sTest.tolist()), 1./c0, dpx,
            #                          cpx, cpy, sx_m, sy_m)
            dist_counter = dist_counter + 1
            #dist = np.asarray(dist_m)

            # Interpolate arrival times at transducer locotions from field 
            # tMatTest = np.zeros(tMat.shape)
            # for cnt in range(nSrc):
            #     tMatTest[:, cnt] = griddata(grid, dist[:, :, cnt].T.flatten(),
            #                                 receivers)
            
            # tMatTest = interpolate(dist.transpose(2, 0, 1).reshape(nSrc, -1), vtx, wts,
            #                        dim=3).T

            resMatTest = tMat - (tMatTest - tMatInc)
            resMatTest[~np.isfinite(resMatTest)] = 0
            
            R, _ = get_grad_2nd(sTest, dpx)
            F, _ = get_grad_1st(sTest, dpx)
            TV, _ = get_grad_tot_var(sTest)
            
            costFnTest = (np.sum(np.sum(resMatTest**2)) + R*eps2 + F*eps
                          + TV*epsTV)
            print('Line search counter: {}, cost function: {}'.format(lineSearchCnt,costFnTest))

            if costFnTest < costFn + alpha*t*mu:
                break
            if lineSearchCnt > 10:
                # Stuck in a rut - reset direction to gradient
                descDir = -grad
                break
            t = t*beta
            
        costFn = costFnTest
        print('Cost function: {}'.format(costFn))
        resMat = resMatTest
        s = sTest
 
        _, gradR = get_grad_2nd(s, dpx)
        _, gradF = get_grad_1st(s, dpx)
        _, gradTV = get_grad_tot_var(s)
        gradR = np.asarray(gradR)
        gradF = np.asarray(gradF)
        gradTV = np.asarray(gradTV)

        grad = ray_project(dpx, cpx, cpy, sx, sy, rx, ry,
                   np.ascontiguousarray(dist), ds, -resMat, angles, sgp,
                   np.ascontiguousarray(s))
        grad = weld_mask.T*(np.asarray(grad.T) + gradR*eps2 + gradF*eps +
                            gradTV*epsTV)
    #tftVel = 1./s
    angles = np.rad2deg(s)
    #eng.quit()
    return angles, gradR, gradF, gradTV
#     figure(imFig)
#     cartimagesc(tftVel,gx,gy)
#     axis image
#     colorbar
#   close(imFig);


