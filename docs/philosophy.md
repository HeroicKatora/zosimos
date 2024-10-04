Initially and for the forseeable future we will require _at least some_ device
and driver supported by any of `wgpu`'s backend. We will try to keep the basic
interface agnostic of that specific though and might offer a pure CPU-based
solution or SIMD acceleration later, essentially emulating the Vulkan API and
forgoing the shader compilation/upload to gpu memory. But then again, we might
expect a general purpose CPU-based drop-in Vulkan implementation on the
platform. That's will be decided based on required effort and timing..

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

## What it is not

¹Such as ImageMagick whose enormous mix of decoding/encoding/computing tasks
and half-baked HW acceleration the author personally views as a crime against
color science accuracy (if your _composition_ library starts out with 'gamma
correction' as an available color operation before physically basaed whitepoint
corrections you don't display much competence in what you or your users are
doing), resource efficient computation, web-server security, and several
software principles (though not against doing a decent job a doing tons of
stuff).

In particular there will be very little IO done by this library. Unless we get
a OS agnostic disk/memory-to-GPU-memory API, in which case we might relax this
to perform some limited amount of pre-declared decoding work in compute shaders
if this leads to overwhelming efficiency gains in terms of saved CPU cycles or
parallelism. Again, we will even then require the caller to very strictly setup
the transfer channel itself such as the binding of File Descriptors, avoid any
operation requiring any direct permissions checks (i.e. use of our process
personality and credentials).
