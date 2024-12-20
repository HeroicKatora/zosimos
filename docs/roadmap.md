# Roadmap / Feature Future

We want documentation on <http://docs.rs/zosimos>.

## Short term

- [x] Functions in the `command` module, stack-based control flow.
- [x] Generics in the `command` module, monomorphizing when lowering into a
  `Program`.
- [ ] A better testing approach and an exhaustive support matrix.
  - [ ] Which color spaces are supported.
  - [ ] Validation for the discovered resource requirements and the `Limits`
    structure returned from a program evaluation. Since this is used during the
    selection of an adapter and device, the mechanism for propagating these
    plays a crucial role in the interoperability.
  - [ ] More devices to verify the hashing exhaustiveness. Or changes that make
    it redundant. Can we eliminate all device-dependent outputs by construction
    without disastrously compromising the performance?
- [.] More operators based on a non-statically typed operator infrastructure
  including binary and unary operators just derived from their shader.
  Investigate the ergonomics by changing some explicitly enumerated built-ins
  that could be runtime.
  - [ ] Proper trait bounds on parameters.
- [ ] An explicit treatment of the 'working color space'. Some operators that
  operate on channels are completely oblivious to whether the image is passed
  in a special channel encoding. For instance, an Oklab may be passed as Lab or
  LCh or even as RGB. Indeed, it may be desired to offer this choice! Also note
  that images with no defined color space may be passed to some images,
  necessitating the 'ad-hoc' ascription of a working color space to images.

  The list of supported working color spaces need not consider all texel
  encodings, indeed it currently is effectively a single entry, `["RGB"]`. But
  this excludes some images from use and with the trait bounds above could be
  neatly expressed for binary operators that need a consistent one for both
  inputs.

## Medium term

- [ ] Render Masks. These would instantly offer a great deal of customization
  to any 'painting' and compositing operation. Unsure if this is a separate
  type with constructor methods; probably. Can they be applied to inputs or on
  some operations? Clear in the mutable `High` IR but not for the SSA form
  that's user facing.
- [ ] Investigate compositing support. Many image operations are currently
  always invoked as a `LoadOp::Clear`, i.e. they only construct a buffer.
  However with the above masking and clipping it would be cheap to apply them
  to part of an image and make a composite at the same time. Is it useful to
  offer this at the surface level or should it be an optimizer result? We need
  clear quantifiable goals here to attemp this question.
- [ ] A fuel system. If one embeds this as-a-service then the costs must be
  apparent. It must be possible to 'interrupt' a pipeline at predictable
  performance critical points. The current operation count based fuel does not
  correspond to any metric the user can understand, it is an implementation
  detail!
- [ ] Support for bounds and capability requirements in function signatures.
  For instance we should be able to determine, after monomorphizing, what
  texture dimensions a program requires. Note a conflict that explicit
  annotations could potentially solve: on the one hand the texture size should
  be hidden from the user. Indeed, it is probably far more useful if we had a
  form of staging buffers / automatically split operations across texture
  windows as a method of handling outsized textures. On the other hand, absence
  of emulation is very important for performance. This behaves a little like
  CPU features except that required capabilities are non-binary.
- [ ] Optimizer passes. This is meant for different layers. For instance the
  encoder at the time of writing (2024-October) will unconditionally load a
  texture from its staging buffer even if the corresponding texture already
  contains the same pixel data due it being unchanged from an earlier load.
- [ ] Runtime hooks on the host. While the library does not do IO (as in
  philosophy) and such hooks serialize the execution unnecessarily, in some
  situations the user may fine cost acceptable for the gain of arbitrarily
  executing intermediate Rust / host / CPU code.
- [ ] Non-Linear control flow, possibly scalar runtime values and a concept of
  dynamic host-side buffers. In particular, a read-back operation that isn't an
  output so that hooks can be written.

## Long term
- [ ] Oversized images. It's no problem for almost arbitrarily large textures
  to be stored as a `wgpu::Buffer` as long as they fit. Working with these
  requires a split of operations across windows but is otherwise
  straightforward for any function so decomposable. Note that this may overlap
  but does not implement external textures, i.e. whose buffer is loaded and
  stored on demand outside the GPU / library memory.
  - This interacts with bounds if the property of a texture being oversized /
    buffer emulated requires trait bound treatment. That is almost surely the
    case to allow operations to fail being instantiated at such types, early.
