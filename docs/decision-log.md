This document is a list of *decisions* that occurred during the design
iteration and implementation phase. It's currently small enough to fit here but
may be split to a larger format once appropriate. The use of this document is
in building understanding in contributors and maintainers for future decisions.
It also enables re-evaluating decisions with little overhead.

## 0 – Template

Decision: An example of decisions is added to this document.
Pro: 
- A rough structure to copy&paste.
Con:
- The document becomes longer, with padding that isn't directly informative.

## 1 – Copy of buffers

Decision: Buffer sizes and copies must be aligned to `4`.
Pro:
- This constraint is required by `wgpu` / common across downlevel APIs for the
  builtin queue.
- Relaxing a constraint at a later point is simpler than introducing one.
- A relaxed, separate version can always be added as long as it is not a
  representational invariant of the binary buffer type.
- Stark performance differences would be similar in effect to a constraint,
  with regards to usability of such parameters in practice, but come with none
  of the diagnostics.
Con:
- The semantics of the surface language are now tied to the implementation.
  This conflicts with the notion of a compiler of an independent language.
