# jax-port
CLI script for porting code using numpy and scipy to their jax equivalent

## Usage
The script only uses the standard library so the should be no further setup needed if you already have python installed
Note that only python 3.9 and later are supported due to the use of ast.unparse
```bash
python jax_port.py -i some_numpy_code.py > some_jax_code.py
```
## Roadmap
- [ ] Support the rest of the special array updates (https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#array-updates-with-other-operations)
- [ ] Port random usage (https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers)
- [ ] Port control flow (https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow)
- [ ] A dynamic mode to ensure we don't break normal list
- [ ] Support non-array input (https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#non-array-inputs-numpy-vs-jax)
- [ ] 
