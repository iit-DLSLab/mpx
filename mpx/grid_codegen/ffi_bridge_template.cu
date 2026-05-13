// Template for the robot-specific GRiD XLA FFI bridge.
//
// A completed bridge must include the generated grid.cuh, flatten leading JAX
// batch dimensions to NUM_TIMESTEPS / num_instances, launch GRiD kernels once
// per batched call, and export these typed FFI handlers:
//
//   extern "C" XLA_FFI_DEFINE_HANDLER_SYMBOL(mpx_grid_step, ...)
//   extern "C" XLA_FFI_DEFINE_HANDLER_SYMBOL(mpx_grid_step_with_derivatives, ...)
//
// The Python backend registers those symbols as:
//   <prefix>_step
//   <prefix>_step_with_derivatives
//
// Keep this file as a starting point for generated bridge code rather than
// editing GRiD's generated CUDA header by hand.
