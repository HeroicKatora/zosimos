# Roadmap / Feature Future

We want documentation on <http://docs.rs/zosimos>.

## Medium term

- [x] Functions in the `command` module, stack-based control flow.
- [x] Generics in the `command` module, monomorphizing when lowering into a
  `Program`.
- [ ] Render Masks. Unsure if this is a separate type with constructor methods;
  probably. Can they be applied to inputs or on some operations? Clear in the
  mutable `High` IR but not for the SSA form that's user facing.
- [ ] Non-Linear control flow, scalar runtime values.
- [ ] More operators based on a non-statically typed operator infrastructure
  including binary and unary operators just derived from their shader.
- [ ] Oversized images. It's no problem for almost arbitrarily large textures
  to be stored as a `wgpu::Buffer` as long as they fit. Working with these
  requires a split of operations across windows but is otherwise
  straightforward for any function so decomposable. Note that this may overlap
  but does not implement external textures, i.e. whose buffer is loaded and
  stored on demand outside the GPU / library memory.
  - This interacts with bounds if the property of a texture being oversized /
    buffer emulated requires trait bound treatment. That is almost surely the
    case to allow operations to fail being instantiated at such types, early.
- [ ] Optimizer passes. This is meant for different layers. For instance the
  encoder at the time of writing (2024-October) will unconditionally load a
  texture from its staging buffer even if the corresponding texture already
  contains the same pixel data due it being unchanged from an earlier load.

## Long term

- [ ] Support for bounds and capability requirements in function signatures.
  For instance we should be able to determine, after monomorphizing, what
  texture dimensions a program requires. Note a conflict that explicit
  annotations could potentially solve: on the one hand the texture size should
  be hidden from the user. Indeed, it is probably far more useful if we had a
  form of staging buffers / automatically split operations across texture
  windows as a method of handling outsized textures. On the other hand, absence
  of emulation is very important for performance. This behaves a little like
  CPU features except that required capabilities are non-binary.
