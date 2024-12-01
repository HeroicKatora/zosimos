# Zosimos

A library for common image operations, so quick and embeddable that you might
barely notice it running. At least that's the goal. The implementation uses
pre-built GPU pipelines, the same you will find used in video games, to wrap an
image manipulation interface more familiar from CPU-based libraries.

Plan: See <docs/roadmap.md>.

Ideas for contributions: See <docs/ideas.md>.

## How to run

This is currently a library. You can run its test suite which will also contain
usage examples for various different editing jobs.

Work-In-Progress: An interactive 'editor' interface is being sketched in
`bin/editor`.

Work-In-Progress: An non-interactive scripting interface that maps the command
operations and whole compile process should be built in `bin/sh`. Here we will
focus on a *batch* use case since re-use of the pipeline is a big computational
advantage. (It remains to be seen if caches can be implemented).

## Project philosophy and goals

> AM/FM
> =====
>
> AM/FM is an engineer's term distinguishing the inevitable clunky real-world
> faultiness of "Actual Machines" from the power-fantasy techno-dreams of
> "Fucking Magic." (Source: Turkey City Lexicon)

For more see <docs/philosophy.md>.

## How to test

_Warning_: The test suite checks for _pixel-accurate results_ by doing a CRC
check on the contents of images. It is somewhat likely that those fail on your
machine as there are allowed differences in the exact floating point math.
This, for example, affects rotated images sampled with `Nearest` pixels as well
as buffer stores relying on the driver to perform `sRGB` encoding,
float-scaling, interpolation of vertex attributes across fragments etc.

Run on the first time, then run:

```
ZOSIMOS_BLESS=1 cargo test --release
cargo test
```

As an added benefit, the first call will produce a debug version of all test
results in the form of `png` images within the [`tests/debug`](./tests/debug)
folder.

## How it works

This project never had the goals of being a cairo alternative. Learning from
graphics interfaces, we rely on a declarative and ahead-of-time specification
of your operations pipeline. We also assume that any resources (except
temporary inputs and outputs byte buffers) are owned by a library object. This
has the advantage that we might reuse memory, have intermediate results that
are never backed by CPU accessible memory, may change certain layouts and
sampling descriptors on the fly, and can plan execution steps and resource
utilization transparently.
