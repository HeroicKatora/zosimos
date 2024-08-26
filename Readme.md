# Stealth-paint

A library for common image operations, so quick and embeddable that you might
barely notice it running. At least that's the goal. The main idea is to use
pre-built GPU pipelines, the same you will find used in video games, but to
wrap it into an interface more familiar from CPU-based methods.¹

## How to run

_Warning_: The test suite checks for _pixel-accurate results_ by doing a CRC
check on the contents of images. It is somewhat likely that those fail on your
machine as there are allowed differenced in the exact floating point math.
This, for example, affects rotated images sampled with `Nearest` pixels as well
as buffer stores relying on the driver to perform `sRGB` encoding,
float-scaling, interpolation of vertex attributes across fragments etc.

Run on the first time, then run:

```
STEALTH_PAINT_BLESS=1 cargo test --release
cargo test
```

As an added benefit, the first call will produce a debug version of all test
results in the form of `png` images within the [`tests/debug`](./tests/debug)
folder.

Otherwise, see the documentation on: <http://docs.rs/>

## Roadmap / Feature Future

#### Medium term

- [ ] Render Masks. Unsure if this is a separate type with constructor methods;
  probably. Can they be applied to inputs or on some operations? Clear in the
  mutable `High` IR but not for the SSA form that's user facing.
- [ ] Rename; some memorable named based on a pun for magick, color, image,
  Lucifer and witches, and parallel processing?
- [ ] More operators based on a non-statically typed operator infrastructure
  including binary and unary operators just derived from their shader.

#### Long term

- [ ] Non-Linear control flow, scalar runtime values.
- [ ] Functions in the `command` module, stack-based control flow.
- [ ] Generics in the `command` module, monomorphizing when lowering into a
  `Program`.


## Project goals

> AM/FM
> AM/FM is an engineer's term distinguishing the inevitable clunky real-world faultiness of "Actual Machines" from the power-fantasy techno-dreams of "Fucking Magic." (Source: Turkey City Lexicon)

Without naming specific other solutions, relying on magic for your image
processing needs invites the risk of many CVEs, a world of painful
configuration, and overall slowness. Avoid all of this by relying on safe Rust,
nice embedding, hardware acceleration and an optimizing execution engine.

Initially and for the forseeable future we will require _at least some_ device
and driver supported by any of `wgpu`'s backend. We will try to keep the basic
interface agnostic of that specific though and might offer a pure CPU-based
solution or SIMD acceleration later, essentially emulating the Vulkan API and
forgoing the shader compilation/upload to gpu memory. But then again, we might
expect a general purpose CPU-based drop-in Vulkan implementation on the
platform. That's not yet decided.

The other more interesting point to resolve at some future version is the case
of external memory/pixel container, as necessary for pipelines operation on
images too large for any of the computer's memories.

We will also try to make it 'nearly realtime' (*cough*) in that the main
execution of a program should be free of infinite cycles, instead based on
step-by-step advance (with hopefully bounded time by the underlying driver) and
fuel based methods, as well as providing ahead of time estimates and bounds on
our resource usage. While we're not close to it yet, at least this is part of
the API reasoning and it's worthy of a bug report if some system actively makes
it impossible.

## How it works

This project never had the goals of being a cairo alternative. Learning from
graphics interfaces, we rely on a declarative and ahead-of-time specification
of your operations pipeline. We also assume that any resources (except
temporary inputs and outputs byte buffers) are owned by a library object. This
has the advantage that we might reuse memory, have intermediate results that
are never backed by CPU accessible memory, may change certain layouts and
sampling descriptors on the fly, and can plan execution steps and resource
utilization transparently.

## What it is not

¹Such as ImageMagick whose enormous mix of decoding/encoding/computing tasks
and half-baked HW acceleration the author personally views as a crime against
color science accuracy (if your _composition_ library starts out with 'gamma
correction' as an available color operation you have no idea what you're
doing), resource efficient computation, web-server security, and several
software principles (though not against doing a decent job a doing tons of
stuff).

In particular there will be no IO done by this library. Unless we get a OS
agnostic disk/memory-to-GPU-memory API, in which case we might relax this to
perform some limited amount of pre-declared decoding work in compute shaders if
this leads to overwhelming efficiency gains in terms of saved CPU cycles or
parallelism. Again, we will even then require the caller to very strictly setup
the transfer channel itself such as the binding of File Descriptors, avoid any
operation requiring any direct permissions checks (i.e. use of our process
personality and credentials).

# Future ideas

I'm really grateful if you pick any of the below. Feel free to grab one that
tickles your interest. (And see 'Project goals' for less concrete tasks).

## Cool stuff with WASM as computation

ImageMagick offers some generic formula application with some custom language.
I don't want to do this but see the purpose of an image transformation language
that is not specific to any API. Why not use WASM for this? This would allow
writing and compiling such code in any _other_ (runtime-free) language first.
If you feel like inventing such a language and writing a compiler to SPIR-V for
it then feel free to make it and then to PR it.

## Similarity measures

- <https://en.wikipedia.org/wiki/Structural_similarity>
- Perceptual hashes

## Edge detection

- <https://en.wikipedia.org/wiki/Edge_detection>
- <https://en.wikipedia.org/wiki/Hough_transform>

## Noise constructors

Just, any. Although keep in mind to use a deterministic method in order to stay
reproducible. Ever wanted to create mega-bytes of pseudo-randomness with a
fragment shader call?
