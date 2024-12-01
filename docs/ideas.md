I'm grateful if you pick any of the issues documented to explore, implement and
demonstrate them. Feel free to grab one that tickles your interest. However,
these are vague concepts that I do not see myself focussing on directly. For
contributions keep in mind maintainability, i.e. try not to introduce anything
incongruent to the project goals and prefer high-level primitives to deep
integration / changes to the coammnd structure.

## Cool stuff with WASM as computation

ImageMagick offers some generic formula application with some custom language.
I don't want to do this but see the purpose of an image transformation language
that is not specific to any API. Why not use WASM for this? This would allow
writing and compiling such code in any _other_ (runtime-free) language first.
If you feel like inventing such a language and writing a compiler to SPIR-V for
it then feel free to make it and then to PR it.

After a bunch of thought, it may be best to ship this as a separate binary
target integrated with the editor and not in the main library component. This
also comes with all caveats of host-side hooks *except* for the platform
dependency.

## Similarity measures

- <https://en.wikipedia.org/wiki/Structural_similarity>
- Perceptual hashes. These might even replace parts of the test suite.

## Edge detection

- <https://en.wikipedia.org/wiki/Edge_detection>
  - Some algorithms already implemented
- <https://en.wikipedia.org/wiki/Hough_transform>
  - Parallel count into buckets relies on atomic writes to a target.
    - We probably want to parameterize that with an explicit result image. That
      is, in the algorithm a line incidence is recorded to buckets. There is no
      best method for defining buckets though and the grid-methods are inferior
      to more generic methods that have space-efficient representations of
      relatively sparse but deep target bucket coordinates (i.e. K-D tree based
      methods).
    - The classic grid method guarantees we have some known box of parameter
      buckets to consider. A "freeform" algorithm where each output box defines
      its own parameter ranges to track would not allow this.

## Noise constructors

Just, any. Although keep in mind to use a deterministic method in order to stay
reproducible. Ever wanted to create mega-bytes of pseudo-randomness with a
fragment shader call?

## Tooling

The project is a programming language, and a compiler, and an interpreter. All
of these would usually come with tooling to assist a developer working with the
various stages. For many instances, the tooling on the host side can be
repurposed but lacks most of the specifics that would make this truly
successful. 

For instance, `renderdoc` will pick up on any (single device) render pipeline.
Yet this will lack any reference to the original command buffer and the
compilation process which resulted in the issued GPU command buffers. This
effectively makes actual root cause analysis very difficult. The library tries
to mitigate this by collecting information during the compile process but the
interfaces for working with those are very ad-hoc instead of structured and
well-designed.

Also we should clarify what the surface language for usage should be and then
provide a form of tree-sitter or language-server based support for it.
