use crate::buffer::Descriptor;

/// Describes the interface of a compiled `Program` (which is always monomorphic).
///
/// The indices of inputs and outputs refer to the local indices within the program's source
/// commands. They are not the same as a `Register`, which is an index into the commands that
/// refers to the command that introduced the input or output.
///
///
#[derive(Default, PartialEq)]
pub struct Function {
    pub(crate) inputs: Vec<Descriptor>,
    pub(crate) outputs: Vec<Descriptor>,
    pub(crate) capabilities: Capabilities,
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
