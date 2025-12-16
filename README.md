This is a quick-and-dirty repro of a vectorization fail. It should work with LLVM 21 and 22-git; the nix flake uses LLVM 21.

To compile the baseline version without vectorization, run the compile_baseline script. 
On LLVM 22, you will need to add `modify-public-functions=1` to the options of the `--buffer-results-to-out-params` pass.


mnist.mlir is the Flax Linen MNIST CNN example exported to StableHLO and then lowered to Linalg.
I've done this already, as StableHLO needs to be compiled from source.

If you want to reproduce the export:

```python
# After you've trained your model, and defined a fordward function to be exported:
state: flax.training.train_state.TrainState = ...
def inference_fn(x):
    return state.apply_fn({'params': state.params}, x, mutable=False)
    
# This should be whatever the shape of the input is
input_sample = jnp.ones([1, 28, 28, 1])

# Now we can export StableHLO
with open("mnist_stablehlo.mlir", "w") as f:
    # Nominally, you are supposed to use jax.export for this but I've had better luck with this method
    f.write(jax.jit(inference_fn).lower(input_sample).as_text())
```
Then you can convert the resulting StableHLO to Linalg with your compiled `stablehlo-opt`:
```sh
stablehlo-opt mnist_stablehlo.mlir --inline --stablehlo-target-independent-optimization --stablehlo-legalize-to-linalg -o mnist.mlir
```
Strictly speaking, the inlining and optimization are not necessary, but they can significantly reduce the IR size and generally make your life much easier.
