//! ## Quick guide
//!
//! 1. Create a Pool of image resources
//! 2. Create a CommandBuffer for describing the operations
//! 3. Fill the resource pool with inputs
//! 4. Enqueue commands to the buffer and compile it
//! 5. Prepare execution of the buffer with your resource pool
//! 6. Retrieve results
//!
//! For steps 3 and 6, input and output, you might find the `image` and `image-canvas` crates
//! quite helpful for dealing with image formats and describing pre-existing buffers.

mod command;
mod pool;
mod program;
mod run;
mod shaders;
