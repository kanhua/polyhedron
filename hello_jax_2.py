from polyhedron_jax import sign_jax, vertex_sign_jnp, vertex_sign_jnp_loop, \
    edge_sign_jnp, triangle_sign_jnp, triangle_chain_jax
import jax.numpy as jnp
import jax
from polyhedron import sign, vertex_sign, edge_sign, triangle_sign, triangle_chain
import time

seed = 1632
key = jax.random.PRNGKey(seed)

# in this test, test_P, test_Q and testO all have different values
test_P = jax.random.uniform(key, (100000, 3), minval=-5, maxval=2)
test_Q = jax.random.uniform(key, (100000, 3), minval=3, maxval=8)
test_R = jax.random.uniform(key, (100000, 3), minval=10, maxval=13)
test_O = jax.random.uniform(key, (3,), minval=-10, maxval=-8)
test_o = list(test_O)

print("running with jnp")

t = time.process_time()
jnp_result = triangle_chain_jax(test_P, test_Q, test_R, test_O)
elapsed_time = time.process_time() - t
print("elapsed time with JAX: {}".format(elapsed_time))

print("running with pure python")
t = time.process_time()
old_code_result = jnp.array(
    [triangle_chain(list(test_P[idx, :]), list(test_Q[idx, :]), list(test_R[idx, :]), test_o) for idx in
     range(test_P.shape[0])])
elapsed_time = time.process_time() - t
print("elapsed time with pure python: {}".format(elapsed_time))

assert jnp.alltrue(jnp_result == old_code_result)
