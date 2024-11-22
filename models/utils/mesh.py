#!/usr/bin/env python3

import numpy as np
import skimage.measure
import torch

import os, pdb
import open3d as o3d
from skimage.morphology import flood
from skimage.segmentation import find_boundaries
from rdp import rdp
from triangulator.ear_clipping_method import triangulate


# N: resolution of grid; 256 is typically sufficient 
# max batch: as large as GPU memory will allow
# shape_feature is either point cloud, mesh_idx (neuralpull), or generated latent code (deepsdf)
def create_mesh(
    model, 
    shape_feature, 
    filename, 
    sem_labels, 
    N=256, 
    max_batch=1000000, 
    level_set=0.0, 
    occupancy=False, 
    point_cloud=None, 
    from_plane_features=False, 
    from_pc_features=False, 
    render_scene=True,
    wall_remesh=False,
):

    model.eval()
    n_labels = len(sem_labels.keys())

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    cube = create_cube(N, n_labels)
    cube_points = cube.shape[0]
    head = 0
    while head < cube_points:
        
        query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
        
        # inference defined in forward function per pytorch lightning convention
        if from_plane_features:
            pred_sdf = model.forward_sdf(shape_feature.cuda(), query.cuda()).detach().cpu()
        else:
            pred_sdf = model(shape_feature.cuda(), query.cuda()).detach().cpu()

        cube[head : min(head + max_batch, cube_points), 3:] = pred_sdf.squeeze()
            
        head += max_batch
    # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
    sdf_values = cube[:, 3:] - 0.5 if occupancy else cube[:, 3:] 
    for i in range(n_labels):
        sdfv = sdf_values[:, i].reshape(N,N,N)
            
        convert_sdf_samples_to_ply(
            sdfv.data,
            voxel_origin,
            voxel_size,
            os.path.join(filename, sem_labels[str(i)] + ".ply"),
            level_set,
            wall_remesh=wall_remesh and sem_labels[str(i)] == "mesh",
        )
    if render_scene:
        sdfv = sdf_values.min(-1)[0].reshape(N, N, N)
        convert_sdf_samples_to_ply(
                sdfv.data,
                voxel_origin,
                voxel_size,
                os.path.join(filename, "scene.ply"),
                level_set
            )


# create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
def create_cube(N, n_labels):

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3+n_labels)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples



def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level_set=0.0,
    wall_remesh=False,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    # use marching_cubes_lewiner or marching_cubes depending on pytorch version 
    # print(np.min(numpy_3d_sdf_tensor), np.max(numpy_3d_sdf_tensor))
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print("skipping {}; error: {}".format(ply_filename_out, e))
        return

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # refer to paper (Fig.14)
    if wall_remesh:
        print("Remeshing wall...")
        bottom = mesh_points[:,1].min()
        top = mesh_points[:,1].max()
        reso = pytorch_3d_sdf_tensor.shape[0]

        # create floor map
        floor = torch.zeros([1,1,reso, reso]).float().cuda()
        floor.requires_grad_(True)  

        # map wall vertices onto floor 
        verts_t = torch.from_numpy(mesh_points).float().cuda()
        torch.nn.functional.grid_sample(
            floor, 
            torch.stack([verts_t[:,0], verts_t[:,2]], 1).unsqueeze(0).unsqueeze(0).contiguous(),
            align_corners=True,
        ).sum().backward()
        cnt = floor.grad[0,0].detach().cpu().numpy()

        # get the wall layout map: cnt
        thres = 0
        cnt[cnt<=thres] = 0
        cnt[cnt>thres] = 1
        cnt = cnt.astype(np.uint8)

        # deal with boundary condition
        cnt_x, cnt_y = np.where(cnt==1)
        if np.sum(cnt_x == 0)>0:
            m = cnt_x == 0
            cnt[cnt_x[m], cnt_y[m]] = 0
            cnt[cnt_x[m]+1, cnt_y[m]] = 1
        if np.sum(cnt_x == reso-1)>0:
            m = cnt_x == reso-1
            cnt[cnt_x[m], cnt_y[m]] = 0
            cnt[cnt_x[m]-1, cnt_y[m]] = 1
        cnt_x, cnt_y = np.where(cnt==1)
        if np.sum(cnt_y == 0)>0:
            m = cnt_y == 0
            cnt[cnt_x[m], cnt_y[m]] = 0
            cnt[cnt_x[m], cnt_y[m]+1] = 1
        if np.sum(cnt_y == reso-1)>0:
            m = cnt_y == reso-1
            cnt[cnt_x[m], cnt_y[m]] = 0
            cnt[cnt_x[m], cnt_y[m]-1] = 1

        ### boundary extraction
        mask = flood(cnt, (0, 0))
        bounds = find_boundaries(mask, mode="outer")
        bx, by = np.where(bounds==True)

        # order the boundary
        def search_path(sx, sy, bounds):
            assert bounds[sx, sy] == True 
            xx, yy = [sx], [sy]
            bounds[xx,yy] = False
            while np.sum(bounds)>0:
                if bounds[sx+1,sy]:
                    sx = sx + 1
                elif bounds[sx-1,sy]:
                    sx = sx - 1
                elif bounds[sx,sy+1]:
                    sy = sy + 1
                elif bounds[sx,sy-1]:
                    sy = sy - 1
                elif bounds[sx+1,sy+1]:
                    sx = sx + 1
                    sy = sy + 1
                elif bounds[sx-1,sy+1]:
                    sx = sx - 1
                    sy = sy + 1
                elif bounds[sx+1,sy-1]:
                    sx = sx + 1
                    sy = sy - 1
                elif bounds[sx-1,sy-1]:
                    sx = sx - 1
                    sy = sy - 1
                else:
                    leftxy = np.where(bounds==True)
                    print(f"{leftxy} not be added!!!")
                    return xx, yy, len(leftxy[0])
                bounds[sx,sy] = False
                xx.append(sx)
                yy.append(sy)
            return xx, yy, len(np.where(bounds==True)[0])

        vx, vy, left_N = search_path(bx[0], by[0], bounds.copy())

        # contour simplification
        def contour_simplification(vx, vy):
            arr = np.stack([vx,vy], 1)
            mask = rdp(arr, algo="iter", return_mask=True, epsilon=4.0)
            res = arr[mask]
            return res[:, 0], res[:, 1]

        vx, vy = contour_simplification(np.array(vx), np.array(vy))

        # wall constraction
        N = len(vx)
        Nid = np.arange(N-1)
        v1 = np.stack([vy / (reso-1) * 2 - 1, np.ones([N])*bottom, vx / (reso-1) * 2 - 1], 1)
        v2 = np.stack([vy / (reso-1) * 2 - 1, np.ones([N])*top, vx / (reso-1) * 2 - 1], 1)
        t1 = np.stack([Nid, Nid+1, Nid+N], 1)
        t2 = np.stack([Nid+1,Nid+N+1,Nid+N], 1)
        vv = np.concatenate([v1,v2])
        tt = np.concatenate([t1,t2,np.array([[N-1,0,2*N-1],[0,N,2*N-1]]).astype(np.int64)])
        
        # floor constraction
        poly = np.stack([vx, vy], 1)
        poly = tuple(tuple(p) for p in poly)
        poly_t = triangulate(poly)

        coord2pid = {}
        for i in range(N):
            coord2pid[poly[i]] = i 

        floor_t = []
        for t in poly_t:
            floor_t.append([coord2pid[t[0]], coord2pid[t[1]], coord2pid[t[2]]])
        floor_t = np.array(floor_t).astype(np.int64)

        # trianglularization
        wall_mesh = o3d.geometry.TriangleMesh()
        wall_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate([vv]))
        wall_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate([tt, floor_t]))
        o3d.io.write_triangle_mesh(ply_filename_out, wall_mesh)

    else:
        num_verts = mesh_points.shape[0]
        num_faces = faces.shape[0]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(ply_filename_out, mesh)
