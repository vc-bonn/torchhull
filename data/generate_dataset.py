from __future__ import annotations

import pathlib

import moderngl
import numpy as np
import scipy
import torch
import trimesh
from PIL import Image


def perspective(
    fovy: float,
    aspect: float,
    near: float,
    far: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0 / (np.tan(fovy / 2.0) * aspect), 0.0, 0.0, 0.0],
            [0.0, 1.0 / np.tan(fovy / 2.0), 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )


def rotate(
    angle: float,
    x: float,
    y: float,
    z: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rotvec = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z])) * angle

    rotmat = np.eye(4)
    rotmat[0:3, 0:3] = scipy.spatial.transform.Rotation.from_rotvec(rotvec).as_matrix()

    return torch.from_numpy(rotmat).to(dtype=dtype, device=device)


def translate(
    x: float,
    y: float,
    z: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def generate_random_camera(
    fovy: float,
    near: float,
    far: float,
    height: int,
    width: int,
    camera_origin_distance: float,
    camera_position_noise_std: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection_matrix = perspective(fovy, width / height, near, far, dtype, device)

    random_axis = scipy.stats.uniform_direction.rvs(3)
    rng = np.random.default_rng(1337)
    random_angle = rng.uniform(0, np.pi)
    random_translate = rng.normal(0, camera_position_noise_std, size=[3])

    random_r = rotate(random_angle, random_axis[0], random_axis[1], random_axis[2], dtype, device)
    random_t = translate(random_translate[0], random_translate[1], random_translate[2], dtype, device)

    view_matrix = translate(0, 0, -camera_origin_distance, dtype, device) @ random_t @ random_r

    return projection_matrix, view_matrix


def render_masks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    transforms: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ctx = moderngl.create_context(
        standalone=True,
        settings={
            "backend": "egl",
        },
    )

    prog = ctx.program(
        vertex_shader="""
                #version 330
                uniform mat4 transforms;
                in vec3 in_vert;
                out vec3 color;
                void main() {
                    gl_Position = transforms * vec4(in_vert, 1.0);
                    color = vec3(1.0);
                }
            """,
        fragment_shader="""
                #version 330
                in vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
    )

    vbo = ctx.buffer(vertices.to(torch.float32).flatten().cpu().numpy())
    ibo = ctx.buffer(faces.to(torch.int32).flatten().cpu().numpy())
    vao = ctx.vertex_array(
        prog,
        [
            (vbo, "3f", "in_vert"),
        ],
        index_buffer=ibo,
        index_element_size=4,
    )
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)])
    fbo.use()

    masks = torch.empty([transforms.shape[0], height, width, 1], dtype=dtype, device=device)
    for i in range(transforms.shape[0]):
        prog["transforms"].write(
            transforms[i].cpu().numpy().transpose().copy()
        )  # Need transpose: row-major (numpy) --> col-major (OpenGL)
        ctx.clear()
        vao.render(moderngl.TRIANGLES)

        data = fbo.read(components=3)
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

        masks[i, :, :, 0] = torch.from_numpy(image[:, :, 0] / 255).to(dtype=dtype, device=device)

    return masks


def generate_dataset(
    mesh_file: pathlib.Path,
    number_cameras: int = 30,
    fovy: float = np.deg2rad(60.0),
    near: float = 0.1,
    far: float = 1000.0,
    height: int = 1080,
    width: int = 1920,
    camera_origin_distance: float = 2.0,
    camera_position_noise_std: float = 0.1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),  # noqa: B008
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mesh = trimesh.load_mesh(mesh_file)
    vertices = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64, device=device)

    boundingbox = torch.from_numpy(mesh.bounds.copy()).to(dtype=vertices.dtype, device=device)
    extents = torch.from_numpy(mesh.extents.copy()).to(dtype=vertices.dtype, device=device)
    vertices = (vertices - torch.mean(boundingbox, dim=0)) / torch.norm(extents / 2.0)

    projection_matrices = torch.empty([number_cameras, 4, 4], dtype=dtype, device=device)
    view_matrices = torch.empty([number_cameras, 4, 4], dtype=dtype, device=device)
    for i in range(number_cameras):
        projection_matrices[i, :, :], view_matrices[i, :, :] = generate_random_camera(
            fovy,
            near,
            far,
            height,
            width,
            camera_origin_distance,
            camera_position_noise_std,
            dtype=dtype,
            device=device,
        )
    transforms = projection_matrices @ view_matrices

    masks = render_masks(
        vertices,
        faces,
        transforms,
        height,
        width,
        dtype=dtype,
        device=device,
    )

    return projection_matrices, view_matrices, masks


def main() -> None:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    output_dir = pathlib.Path(__file__).parents[1] / "data" / "cache"

    file = "Armadillo.ply"

    output_dir.mkdir(exist_ok=True)

    projection_matrices, view_matrices, masks = generate_dataset(
        mesh_file=data_dir / file,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    transforms = projection_matrices @ view_matrices

    for i, mask in enumerate(masks):
        Image.fromarray(255.0 * mask.squeeze().cpu().numpy()).convert("L").save(output_dir / f"mask_{str(i)}.png")
    torch.save(masks.cpu().numpy(), output_dir / "masks.pt")
    torch.save(view_matrices.cpu().numpy(), output_dir / "view_matrices.pt")
    torch.save(projection_matrices.cpu().numpy(), output_dir / "projection_matrices.pt")
    torch.save(transforms.cpu().numpy(), output_dir / "transforms.pt")


if __name__ == "__main__":
    main()
