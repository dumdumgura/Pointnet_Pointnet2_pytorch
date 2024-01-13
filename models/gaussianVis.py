import numpy as np
import open3d as o3d

def createBatchEllipsoids(n, scale_constants, centers, semi_axes, rotation_matrices):
    resolution = 25  # Number of points on each axis

    # Create points on the ellipsoid's surface
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)

    # Adjust for broadcasting
    cos_u, sin_u = cos_u[np.newaxis, np.newaxis, ...], sin_u[np.newaxis, np.newaxis, ...]
    cos_v, sin_v = cos_v[np.newaxis, np.newaxis, ...], sin_v[np.newaxis, np.newaxis, ...]

    # Scale and semi-axes adjustment
    semi_axes = np.array(semi_axes) * scale_constants[:, np.newaxis]
    semi_axes = semi_axes[:, :, np.newaxis, np.newaxis]

    # Create ellipsoids
    x = semi_axes[:, 0] * cos_u * sin_v
    y = semi_axes[:, 1] * sin_u * sin_v
    z = semi_axes[:, 2] * cos_v

    # Reshape for points
    points = np.stack([x, y, z], axis=-1)
    points = points.reshape(n, -1, 3)  # Flatten resolution dimensions

    # Apply rotation and translation
    points = np.einsum('nij,nkj->nki', rotation_matrices, points)  # Apply rotation
    points += centers[:, np.newaxis, :]  # Translation

    # Create meshes for each ellipsoid
    meshes = []
    for i in range(n):
        # Create triangles for each ellipsoid
        triangles = []
        for j in range(resolution - 1):
            for k in range(resolution - 1):
                p1 = j * resolution + k
                p2 = p1 + 1
                p3 = (j + 1) * resolution + k
                p4 = p3 + 1
                triangles.append([p1, p2, p3])
                triangles.append([p2, p4, p3])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points[i])
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        wireFrame = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        wireFrame.paint_uniform_color([0.2, 0.5, 1])  # Color the wireframe
        meshes.append(wireFrame)

    return meshes

def visualizeBatchGaussians(n):
    # Randomly generate parameters for n Gaussians
    scale_constants = np.random.rand(n)
    centers = np.random.rand(n, 3) * 10
    semi_axes = np.random.rand(n, 3) * 5
    euler_angles = np.random.rand(n, 3) * 2 * np.pi

    # Construct rotation matrices for each ellipsoid
    rotation_matrices = np.empty((n, 3, 3))
    for i in range(n):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(euler_angles[i, 0]), -np.sin(euler_angles[i, 0])],
                        [0, np.sin(euler_angles[i, 0]), np.cos(euler_angles[i, 0])]])
        R_y = np.array([[np.cos(euler_angles[i, 1]), 0, np.sin(euler_angles[i, 1])],
                        [0, 1, 0],
                        [-np.sin(euler_angles[i, 1]), 0, np.cos(euler_angles[i, 1])]])
        R_z = np.array([[np.cos(euler_angles[i, 2]), -np.sin(euler_angles[i, 2]), 0],
                        [np.sin(euler_angles[i, 2]), np.cos(euler_angles[i, 2]), 0],
                        [0, 0, 1]])
        rotation_matrix = R_z @ R_y @ R_x
        rotation_matrices[i] = rotation_matrix

    # Create batch ellipsoids
    ellipsoids = createBatchEllipsoids(n, scale_constants, centers, semi_axes, rotation_matrices)
    # scale: n,1
    # center: n,3
    # semi_axes: n,3
    # rot_mat: n,3,3


    # Visualize all ellipsoids
    o3d.visualization.draw_geometries(ellipsoids)

if __name__ == '__main__':
    visualizeBatchGaussians(10)  # Visualize 10 Gaussians