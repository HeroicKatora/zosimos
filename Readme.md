# Zosimos

A library for common image operations, so quick and embeddable that you might
barely notice it running. At least that's the goal. The main idea is to use
pre-built GPU pipelines, the same you will find used in video games, but to
wrap it into an interface more familiar from CPU-based methods.

Planned: See [docs/roadmap.md].

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

## Project philosophy and goals

> AM/FM
> =====
>
> AM/FM is an engineer's term distinguishing the inevitable clunky real-world
> faultiness of "Actual Machines" from the power-fantasy techno-dreams of
> "Fucking Magic." (Source: Turkey City Lexicon)

Without naming specific other solutions, relying on magic for your image
processing needs invites the risk of many CVEs, a world of painful
configuration, and overall slowness. Avoid all of this by relying on safe Rust,
nice embedding, hardware acceleration and an optimizing execution engine.

For more see [docs/philosophy.md].

## How it works

This project never had the goals of being a cairo alternative. Learning from
graphics interfaces, we rely on a declarative and ahead-of-time specification
of your operations pipeline. We also assume that any resources (except
temporary inputs and outputs byte buffers) are owned by a library object. This
has the advantage that we might reuse memory, have intermediate results that
are never backed by CPU accessible memory, may change certain layouts and
sampling descriptors on the fly, and can plan execution steps and resource
utilization transparently.

# Future ideas

I'm grateful if you pick any of the issues documented to explore, implement and
demonstrate them. Feel free to grab one that tickles your interest. (And see
'Project goals' for less concrete tasks). Keep in mind maintainability, i.e.
try not to introduce anything incongruent to the project goals and prefer
high-level primitives to deep integration / changes to the coammdn structure.

See [docs/ideas.md].
