use std::collections::HashMap;
use crate::buffer::Descriptor;

/// Describes the interface of a compiled `Program` (which is always monomorphic).
///
/// The indices of inputs and outputs refer to the local indices within the program's source
/// commands. They are not the same as a `Register`, which is an index into the commands that
/// refers to the command that introduced the input or output.
#[derive(Default, PartialEq)]
pub struct Function {
    pub(crate) inputs: Vec<Descriptor>,
    pub(crate) outputs: Vec<Descriptor>,
    pub(crate) capabilities: Capabilities,
    /// The type of exposed statics.
    /// The location is kept separate because the translation is in full control over it.
    pub(crate) statics: Vec<StaticKind>,
}

/// The dynamic execution requirements.
///
/// One can think of this structure as ISA feature flags, as well as requirements of devices in the
/// environment. Hopefully, in the future, this will also capture such things as auxiliary
/// hardware, sets of extensions (might be user provided and loaded at runtime), and other
/// functionality provided by the execution environment. We might use this to circumvent our own
/// no-IO policy by moving that problem into the realm of the user's responsibility.
#[derive(Default, PartialEq)]
pub struct Capabilities {
    _void: (),
}

impl Function {
    /// Create a function type mapping an empty set to an empty set.
    pub fn new() -> Self {
        Function::default()
    }

    /// An iterator over all input types in order.
    pub fn inputs(&self) -> impl Iterator<Item=&'_ Descriptor> {
        self.inputs.iter()
    }

    /// An iterator over all output types in order.
    pub fn outputs(&self) -> impl Iterator<Item=&'_ Descriptor> {
        self.outputs.iter()
    }
}

/// The type of a `Static`, a program constant that can be adjusted after compilation.
///
/// This is an interned struct that only makes sense in relation to a particular `Function`.
///
/// This identifies the ABI (layout) of a particular piece of data that a shader might used as
/// input. When a program is started then these are uploaded to device buffers and bound to the
/// shader pipelines to serve as inputs.
///
/// One may rewrite the contents of a static in a `Program` or `Executable` given the correct type
/// and its identifier. Of course, this might have pretty unpredictable results so it should only
/// be done with care, or through more sensible interfaces offered by those commands that return
/// the location of their parameters upon request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Static {
    pub(crate) id: usize,
}

/// A builder for Static types.
#[derive(Default, Clone)]
pub(crate) struct StaticArena {
    /// The map from 'node' in (the explored part of) the graph to the index we have named it.
    interner: HashMap<StaticKind, Static>,
    /// The explored types / nodes in the type DAG.
    inner: Vec<StaticKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layouter {
    Std140,
    Std430,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StaticKind {
    Float,
    Double,
    Vec2,
    Mat2(Layouter),
    Mat3(Layouter),
    Mat4(Layouter),
    /// One static, repeated.
    Array {
        base: Static,
        count: u32,
        layout: Layouter,
    },
    /// One value followed by another [and then another…].
    After {
        before: Static,
        after: Static,
        layout: Layouter,
    },
}

impl StaticArena {
    pub fn push(&mut self, kind: StaticKind) -> Static {
        kind.for_dependencies(&mut |Static { id }| {
            assert!(id < self.inner.len());
        });

        let inner = &mut self.inner;
        *self.interner
            .entry(kind)
            .or_insert_with_key(|&kind| {
                let idx = Static { id: inner.len() };
                inner.push(kind);
                idx
            })
    }
}

impl StaticKind {
    pub fn for_dependencies(mut self, ft: &mut dyn FnMut(Static)) {
        self.visit_dependencies(&mut |&mut dep| ft(dep));
    }

    pub fn visit_dependencies(&mut self, ft: &mut dyn FnMut(&mut Static)) {
        match self {
            StaticKind::Array { base, .. } => {
                ft(base)
            }
            StaticKind::After { before, after, .. } => {
                ft(before);
                ft(after);
            }
            _ => {}
        }
    }
}
