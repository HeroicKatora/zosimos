## Cool stuff with WASM as computation

ImageMagick offers some generic formula application with some custom language.
I don't want to do this but see the purpose of an image transformation language
that is not specific to any API. Why not use WASM for this? This would allow
writing and compiling such code in any _other_ (runtime-free) language first.
If you feel like inventing such a language and writing a compiler to SPIR-V for
it then feel free to make it and then to PR it.

After a bunch of thought, it may be best to ship this as a separate binary
target integrated with the editor and not in the main library component.

## Similarity measures

- <https://en.wikipedia.org/wiki/Structural_similarity>
- Perceptual hashes

## Edge detection

- <https://en.wikipedia.org/wiki/Edge_detection>
- <https://en.wikipedia.org/wiki/Hough_transform>
    - We probably want to parameterize that with an explicit result image. That
      is, in the algorithm a line incidence is recorded to buckets. There is no
      best method for defining buckets though and the grid-methods are inferior
      to more generic methods that have space-efficient representations of
      relatively sparse but deep target bucket coordinates (i.e. K-D tree based
      methods).

## Noise constructors

Just, any. Although keep in mind to use a deterministic method in order to stay
reproducible. Ever wanted to create mega-bytes of pseudo-randomness with a
fragment shader call?
