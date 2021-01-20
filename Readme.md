# Stealth-paint

A library for common image operations, so quick and embeddable that you might
barely notice it running.

## Project goals

> AM/FM
> AM/FM is an engineer's term distinguishing the inevitable clunky real-world faultiness of "Actual Machines" from the power-fantasy techno-dreams of "Fucking Magic." (Source: Turkey City Lexicon)

Without naming specific other solutions, relying on magic for your image
processing needs invites the risk of many CVEs, a world of painful
configuration, and overall slowness. Avoid all of this by relying on safe Rust,
nice embedding, hardware acceleration and an optimizing execution engine.

## How it works

This project never had the goals of being a cairo alternative. Learning from
graphics interfaces, we rely on a declarative and ahead-of-time specification
of your operations pipeline. We also assume that any resources (except
temporary inputs and outputs byte buffers) are owned by a library object. This
has the advantage that we might reuse memory, have intermediate results that
are never backed by CPU accessible memory, may change certain layouts and
sampling descriptors on the fly, and can plan execution steps and resource
utilization transparently.
