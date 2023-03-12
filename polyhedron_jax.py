"""
Jax version of polyhedron.py

"""

import jax
import jax.numpy as jnp


def sign(x):
    """
    Return 1 if x is positive, -1 if it's negative, and 0 if it's zero.

    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def sign_jax(x) -> jnp.ndarray:
    sign_x = jnp.where(x > 0, 1, -1)
    is_zero_x = jnp.where(x == 0, 0, 1)

    sign_x = sign_x * is_zero_x

    return sign_x


def vertex_sign(P, O):
    """
    Sign of the vertex P with respect to O, as defined above.

    """
    result = sign(P[0] - O[0]) or sign(P[1] - O[1]) or sign(P[2] - O[2])
    if not result:
        raise ValueError("vertex coincides with origin")
    return result


def vertex_sign_jnp(P, O):
    """
    array version of vertex_sign

    Args:
        P:
        O:

    Returns:

    """
    x = sign_jax(P[:, 0] - O[0])
    y = sign_jax(P[:, 1] - O[1])
    z = sign_jax(P[:, 2] - O[2])

    result_1 = jnp.where(x != 0, x, y)
    result = jnp.where(result_1 != 0, result_1, z)
    return result


def vertex_sign_jnp_loop(P, O):
    x = sign_jax(P[:, 0] - O[0])
    y = sign_jax(P[:, 1] - O[1])
    z = sign_jax(P[:, 2] - O[2])

    result = xyz_or(x, y, z)

    return jnp.array(result)


def xyz_or(x, y, z):
    result = []
    for i in range(x.shape[0]):
        val = x[i] or y[i] or z[i]
        result.append(val)
    return jnp.array(result)


def edge_sign(P, Q, O):
    """
    Sign of the edge PQ with respect to O, as defined above.

    """
    result = (
            sign((P[1] - O[1]) * (Q[0] - O[0]) - (P[0] - O[0]) * (Q[1] - O[1])) or
            sign((P[2] - O[2]) * (Q[0] - O[0]) - (P[0] - O[0]) * (Q[2] - O[2])) or
            sign((P[2] - O[2]) * (Q[1] - O[1]) - (P[1] - O[1]) * (Q[2] - O[2]))
    )
    if not result:
        raise ValueError("vertices collinear with origin")
    return result


def edge_sign_jnp(P, Q, O):
    x = sign_jax((P[:, 1] - O[1]) * (Q[:, 0] - O[0]) - (P[:, 0] - O[0]) * (Q[:, 1] - O[1]))
    y = sign_jax((P[:, 2] - O[2]) * (Q[:, 0] - O[0]) - (P[:, 0] - O[0]) * (Q[:, 2] - O[2]))
    z = sign_jax((P[:, 2] - O[2]) * (Q[:, 1] - O[1]) - (P[:, 1] - O[1]) * (Q[:, 2] - O[2]))

    result_1 = jnp.where(x != 0, x, y)
    result = jnp.where(result_1 != 0, result_1, z)
    #result = jnp.logical_or(x, jnp.logical_or(y, z))
    #result = xyz_or(x, y, z)
    #if jnp.any(result==0):
    #    raise ValueError("vertices collinear with origin")

    return result


def triangle_sign(P, Q, R, O):
    """
    Sign of the triangle PQR with respect to O, as defined above.

    """
    m1_0 = P[0] - O[0]
    m1_1 = P[1] - O[1]
    m2_0 = Q[0] - O[0]
    m2_1 = Q[1] - O[1]
    m3_0 = R[0] - O[0]
    m3_1 = R[1] - O[1]
    result = sign(
        (m1_0 * m2_1 - m1_1 * m2_0) * (R[2] - O[2]) +
        (m2_0 * m3_1 - m2_1 * m3_0) * (P[2] - O[2]) +
        (m3_0 * m1_1 - m3_1 * m1_0) * (Q[2] - O[2]))
    if not result:
        raise ValueError("vertices coplanar with origin")
    return result


def triangle_sign_jnp(P, Q, R, O) -> jnp.ndarray:
    """

    Args:
        P: Nx3
        Q: Nx3
        R: Nx3
        O: scalar

    Returns:

    """
    m1_0 = P[:, 0] - O[0]
    m1_1 = P[:, 1] - O[1]
    m2_0 = Q[:, 0] - O[0]
    m2_1 = Q[:, 1] - O[1]
    m3_0 = R[:, 0] - O[0]
    m3_1 = R[:, 1] - O[1]
    result = sign_jax(
        (m1_0 * m2_1 - m1_1 * m2_0) * (R[:, 2] - O[2]) +
        (m2_0 * m3_1 - m2_1 * m3_0) * (P[:, 2] - O[2]) +
        (m3_0 * m1_1 - m3_1 * m1_0) * (Q[:, 2] - O[2]))

    # TODO: add exception handling
    #if jnp.any(result == 0):
    #    raise ValueError("vertices coplanar with origin")
    return result


def edge_sanity_check(triangles, vertex_positions):
    # Validate: check the combinatorial data.
    edges = set()
    vertices = set()

    for triangle in triangles:
        vertices.update(triangle)
        P, Q, R = triangle
        for edge in ((P, Q), (Q, R), (R, P)):
            if edge[0] == edge[1]:
                raise ValueError("Self edge: {!r}".format(edge))
            if edge in edges:
                raise ValueError("Duplicate edge: {!r}".format(edge))
            edges.add(edge)
    # For each edge that appears, the reverse edge should also appear.
    for P, Q in edges:
        if not (Q, P) in edges:
            raise ValueError("Unmatched edge: {!r}".format((P, Q)))
    # Vertex set should match indices in vertex_positions.
    if vertices != set(range(len(vertex_positions))):
        raise ValueError("Vertex set doesn't match position indices.")


def triangle_chain(v1, v2, v3, origin):
    """
    Return the contribution of this triangle to the winding number.

    Raise ValueError if the face contains the origin.

    """
    v1sign = vertex_sign(v1, origin)
    v2sign = vertex_sign(v2, origin)
    v3sign = vertex_sign(v3, origin)

    face_boundary = 0
    if v1sign != v2sign:
        face_boundary += edge_sign(v1, v2, origin)
    if v2sign != v3sign:
        face_boundary += edge_sign(v2, v3, origin)
    if v3sign != v1sign:
        face_boundary += edge_sign(v3, v1, origin)
    if not face_boundary:
        return 0

    return triangle_sign(v1, v2, v3, origin)


def triangle_chain_jax(v1, v2, v3, origin):
    """
    Return the contribution of this triangle to the winding number.

    Raise ValueError if the face contains the origin.

    """
    v1sign = vertex_sign_jnp(v1, origin)
    v2sign = vertex_sign_jnp(v2, origin)
    v3sign = vertex_sign_jnp(v3, origin)

    face_boundary = 0
    fb1 = jnp.where(v1sign != v2sign, edge_sign_jnp(v1, v2, origin), 0)
    fb2 = jnp.where(v2sign != v3sign, fb1 + edge_sign_jnp(v2, v3, origin), fb1)
    fb3 = jnp.where(v3sign != v1sign, fb2 + edge_sign_jnp(v3, v1, origin), fb2)

    # TODO: resolve this condition
    # if not face_boundary:
    #    return 0
    # return triangle_sign_jnp(v1, v2, v3, origin)

    result = jnp.where(fb3 == 0, 0, triangle_sign_jnp(v1, v2, v3, origin))
    return result


class Polyhedron(object):
    def __init__(self, triangles, vertex_positions):
        """
        Initialize from list of triangles and vertex positions.

        """
        # triangles = jnp.array(triangles)
        # vertex_positions = jnp.array(vertex_positions)

        edge_sanity_check(triangles, vertex_positions)

        # Vertex positions in R^3.
        self.vertex_positions = vertex_positions
        # Indices making up each triangle, counterclockwise
        # around the outside of the face.
        self.triangles = triangles
        self.vertex_positions_jnp = jnp.array(vertex_positions)
        self.triangles_jnp = jnp.array(triangles)
        self.triangle_positions_jnp = jnp.array(list(self.triangle_positions()))
        print(self.triangle_positions_jnp)

    def triangle_positions(self):
        """
        Triples of vertex positions.

        """
        for triangle in self.triangles:
            yield tuple(self.vertex_positions[vx] for vx in triangle)

    def volume(self):
        """
        Return the volume of this polyhedron.

        """
        acc = 0
        for p1, p2, p3 in self.triangle_positions():
            # Twice the area of the projection onto the x-y plane.
            det = ((p2[1] - p3[1]) * (p1[0] - p3[0]) -
                   (p2[0] - p3[0]) * (p1[1] - p3[1]))
            # Three times the average height.
            height = p1[2] + p2[2] + p3[2]
            acc += det * height
        return acc / 6.0

    def winding_number(self, point):
        """Determine the winding number of *self* around the given point.

        """

        winding_num = triangle_chain_jax(self.triangle_positions_jnp[:, 0, :],
                                         self.triangle_positions_jnp[:, 1, :],
                                         self.triangle_positions_jnp[:, 2, :], jnp.array(point))

        winding_number = jnp.sum(winding_num) // 2

        return winding_number

    def winding_number_old(self, point):
        """Determine the winding number of *self* around the given point.

        """
        print([triangle_chain(v1, v2, v3, point) for v1, v2, v3 in self.triangle_positions()])
        return sum(
            triangle_chain(v1, v2, v3, point)
            for v1, v2, v3 in self.triangle_positions()) // 2
