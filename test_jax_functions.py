import pytest

from polyhedron_jax import sign_jax, vertex_sign_jnp, vertex_sign_jnp_loop, \
    edge_sign_jnp, triangle_sign_jnp,triangle_chain_jax
import jax.numpy as jnp
import jax
from polyhedron import sign, vertex_sign, edge_sign, triangle_sign,triangle_chain


def test_sign_jax():
    test_arr = jnp.array([2, -2, 0])
    result_array = sign_jax(test_arr)
    print(result_array)


def test_sign_jax_2():
    seed = 1701
    key = jax.random.PRNGKey(seed)

    rand_int = jax.random.randint(key, (100,), minval=-2, maxval=2)
    result_array = sign_jax(rand_int)
    old_sign_output = jnp.array([sign(s) for s in rand_int])

    assert jnp.alltrue(result_array == old_sign_output)


def test_vertex_sign():
    seed = 1632
    key = jax.random.PRNGKey(seed)

    test_P = jax.random.randint(key, (100, 3), minval=-5, maxval=5)
    test_O = jax.random.randint(key, (3,), minval=-3, maxval=3)
    test_o = list(test_O)
    jnp_result = vertex_sign_jnp(test_P, test_O)

    old_code_result = jnp.array([vertex_sign(list(test_pp), test_o) for test_pp in test_P])
    assert jnp.alltrue(jnp_result == old_code_result)


def test_vertex_sign_2():
    seed = 1934
    key = jax.random.PRNGKey(seed)

    test_P = jax.random.randint(key, (100, 3), minval=1, maxval=5)
    test_O = jax.random.randint(key, (3,), minval=0, maxval=0)
    test_o = list(test_O)
    jnp_result = vertex_sign_jnp_loop(test_P, test_O)

    old_code_result = jnp.array([vertex_sign(list(test_pp), test_o) for test_pp in test_P])
    assert jnp.alltrue(jnp_result == old_code_result)


def test_edge_sign():
    seed = 1632
    key = jax.random.PRNGKey(seed)

    # in this test, test_P, test_Q and testO all have different values
    test_P = jax.random.randint(key, (100, 3), minval=-5, maxval=2)
    test_Q = jax.random.randint(key, (100, 3), minval=3, maxval=8)
    test_O = jax.random.randint(key, (3,), minval=-10, maxval=-8)
    test_o = list(test_O)

    jnp_result = edge_sign_jnp(test_P, test_Q, test_O)

    old_code_result = jnp.array(
        [edge_sign(list(test_P[idx, :]), list(test_Q[idx, :]), test_o) for idx in range(test_P.shape[0])])
    assert jnp.alltrue(jnp_result == old_code_result)


def test_triangle_sign():
    seed = 1632
    key = jax.random.PRNGKey(seed)

    # in this test, test_P, test_Q and testO all have different values
    test_P = jax.random.uniform(key, (100, 3), minval=-5, maxval=2)
    test_Q = jax.random.uniform(key, (100, 3), minval=3, maxval=8)
    test_R = jax.random.uniform(key, (100, 3), minval=10, maxval=13)
    test_O = jax.random.uniform(key, (3,), minval=-10, maxval=-8)
    test_o = list(test_O)

    jnp_result = triangle_sign_jnp(test_P, test_Q, test_R, test_O)

    old_code_result = jnp.array(
        [triangle_sign(list(test_P[idx, :]), list(test_Q[idx, :]), list(test_R[idx, :]), test_o) for idx in
         range(test_P.shape[0])])
    assert jnp.alltrue(jnp_result == old_code_result)


def test_triangle_chain():
    seed = 1632
    key = jax.random.PRNGKey(seed)

    # in this test, test_P, test_Q and testO all have different values
    test_P = jax.random.uniform(key, (100, 3), minval=-5, maxval=2)
    test_Q = jax.random.uniform(key, (100, 3), minval=3, maxval=8)
    test_R = jax.random.uniform(key, (100, 3), minval=10, maxval=13)
    test_O = jax.random.uniform(key, (3,), minval=-10, maxval=-8)
    test_o = list(test_O)

    jnp_result = triangle_chain_jax(test_P, test_Q, test_R, test_O)

    old_code_result = jnp.array(
        [triangle_chain(list(test_P[idx, :]), list(test_Q[idx, :]), list(test_R[idx, :]), test_o) for idx in
         range(test_P.shape[0])])
    assert jnp.alltrue(jnp_result == old_code_result)
