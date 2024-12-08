use std::collections::HashMap;

use serde::Deserialize;
use zosimos::{
    buffer::{Color, Descriptor, Texel, Whitepoint},
    command::{ChromaticAdaptationMethod, Rectangle},
};

use crate::serde_bindings as bindings;

pub struct SourceCode {
    pub functions: HashMap<u64, Function>,
}

/// The name of a variable in a function.
#[derive(Deserialize)]
pub struct VarName(String);

#[derive(Deserialize)]
pub struct RegisterName(String);

/// A function, a sequence of SSA-like definitions of image and buffer contents.
#[derive(Deserialize)]
pub struct Function {
    /// Type parameters that the caller must define.
    pub generics: Vec<Generic>,
    /// Inputs for this function.
    ///
    /// Not all types are allowed as inputs but images.
    pub parameter: Vec<Parameter>,
    /// The sequence of definitions. Each place follows a single-static-assignment but variables
    /// can be shadowed. So we're building a directed acyclic graph of operations here.
    pub body: Vec<Statement>,
}

#[derive(Deserialize)]
pub struct Generic {
    pub ty_name: VarName,
}

/// Parameters only declare the formal available inputs, i.e. its signature to calls. You'll need
/// to later consume them into actual variables by `Input`.
#[derive(Deserialize)]
pub struct Parameter {
    /// The name to use when referring to this parameter and its type.
    pub parameter: VarName,
}

#[derive(Deserialize)]
pub struct Statement {
    /// The name with which to utilize the state in computation.
    pub var: VarName,
    pub kind: StatementKind,
}

macro_rules! command_fn {
    (
        $(#[$attr:meta])*
        enum $name:ident {
            $(
                $variant:ident {
                    $($(#[$arg_attr:meta])* $arg:ident : $ty:ty),*
                    $(,)?
                }
            ),*
            $(,)?
        }
    ) => {
        $(#[$attr])*
        enum $name {
            $(
                $variant {
                    $($(#[$arg_attr])* $arg: $ty),*
                }
            ),*
        }
    };
}

/// FIXME: derive-like against the impl of `CommandBuffer`.
#[derive(Deserialize)]
pub enum StatementKind {}

command_fn! {
    /// An expression to append to a function block within a statement.
    #[derive(Deserialize)]
    enum StatementOp {
        Input {
            desc: bindings::Descriptor,
        },
        Crop {
            source: RegisterName,
            #[serde(with = "bindings::RectangleU32")]
            rect: Rectangle,
        },
        ColorConvert {
            source: RegisterName,
            #[serde(with = "bindings::Color")]
            color: Color,
            #[serde(with = "bindings::Texel")]
            texel: Texel,
        },
        ChromaticAdaptation {
            source: RegisterName,
            #[serde(with = "bindings::ChromaticAdaptationMethod")]
            method: ChromaticAdaptationMethod,
            #[serde(with = "bindings::Whitepoint")]
            target: Whitepoint,
        },
    }
}
