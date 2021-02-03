use crate::program::{CompileError, Program};
use crate::buffer::{Color, Descriptor};

/// A reference to one particular value.
pub struct Register(usize);

/// One linear sequence of instructions.
///
/// The machine model is a single basic block in SSA where registers are strongly typed with their
/// buffer descriptors.
///
/// *Why not a … stack machine*: The author believes that stack machines are a poor fit for image
/// editing in general. Their typical core assumption is that a) registers have the same size b)
/// copying them is cheap. Neither is true for images.
///
/// *Why not a … mutable model*: Because this would complicate the tracking of types. Due to the
/// lack of loops there is little reason for mutability anyways. If you wrap the program in a loop
/// to simulate branches yourself then each `launch` provides the opportunity to rebind the images
/// or bind an image to an output, where it can be mutated.
///
/// The strict typing and SSA-liveness analysis allows for a clean analysis of required temporary
/// resources, re-usage of those as an internal optimization, and in general simple analysis and
/// verification.
pub struct CommandBuffer {
    ops: Vec<Op>,
}

enum Op {
    /// i := in()
    Input {
        desc: Descriptor,
    },
    /// out(src)
    /// where type(src) = desc
    ///
    /// WIP: and is_cpu_type(desc)
    /// for the eventuality of gpu-only buffer layouts.
    Output { 
        src: Register,
        desc: Descriptor,
    },
    /// i := op()
    /// where type(i) = desc
    Construct {
        desc: Descriptor,
        op: ConstructOp,
    },
    /// i := unary(src)
    /// where type(i) =? Op[type(src)]
    Unary {
        src: Register,
        op: UnaryOp,
    },
    /// i := binary(lhs, rhs)
    /// where type(i) =? Op[type(lhs), type(rhs)]
    Binary {
        lhs: Register,
        rhs: Register,
        op: BinaryOp,
    },
}

enum ConstructOp {
    // TODO: can optimize this repr for the common case.
    Solid(Vec<u8>),
}

enum UnaryOp {
    /// Op = id
    Affine(Affine),
    /// Op = id
    Crop(Rectangle),
    /// Op(color)[T] = T[.color=color]
    /// And color needs to be 'color compatible' with the prior T (see module).
    ColorConvert(Color),
}

enum BinaryOp {
    /// Op[T, U] = T
    /// where T = U
    Inscribe,
}

/// A rectangle in `u32` space.
/// It's describe by minimum and maximum coordinates, inclusive and exclusive respectively. Any
/// rectangle where the order is not correct is interpreted as empty. This has the advantage of
/// simplifying certain operations that would otherwise need to check for correctness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rectangle {
    pub x: u32,
    pub y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

#[non_exhaustive]
pub enum Blend {
    Alpha,
}

pub struct Affine {
    transformation: [f32; 9],
}

pub struct CommandError {
    type_err: bool,
}

impl CommandBuffer {
    /// Declare an input.
    ///
    /// Inputs MUST later be bound from the pool during launch.
    pub fn input(&mut self, _: Descriptor) -> Register {
        todo!()
    }

    /// Select a rectangular part of an image.
    pub fn crop(&mut self, src: Register, rect: Rectangle)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// Create an image with different color encoding.
    pub fn color_convert(&mut self, src: Register, color: Color)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// Embed this image as part of a larger one.
    pub fn inscribe(&mut self, below: Register, _: Rectangle, above: Register)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// Overlay this image as part of a larger one, performing blending.
    pub fn blend(&mut self, below: Register, _: Rectangle, above: Register, _: Blend)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// A solid color image, from a descriptor and a single texel.
    pub fn solid(&mut self, _: Descriptor, _: &[u8])
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// An affine transformation of the image.
    pub fn affine(&mut self, src: Register, _: Affine)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// Declare an output.
    ///
    /// Outputs MUST later be bound from the pool during launch.
    pub fn output(&mut self, src: Register, _: Descriptor)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    pub fn compile(&self) -> Result<Program, CompileError> {
        todo!()
    }
}

impl Rectangle {
    /// A rectangle at the origin with given width (x) and height (y).
    pub fn with_width_height(width: u32, height: u32) -> Self {
        Rectangle { x: 0, y: 0, max_x: width, max_y: height }
    }

    /// The apparent width.
    pub fn width(self) -> u32 {
        self.max_x.saturating_sub(self.x)
    }

    /// The apparent height.
    pub fn height(self) -> u32 {
        self.max_y.saturating_sub(self.y)
    }

    /// Return true if this rectangle fully contains `other`.
    pub fn contains(self, other: Self) -> bool {
        self.x <= other.x && self.y <= other.y && {
            // Offsets are surely non-wrapping.
            let offset_x = other.x - self.x;
            let offset_y = other.y - self.y;
            let rel_width = self.width().checked_sub(offset_x);
            let rel_height = self.height().checked_sub(offset_y);
            rel_width >= Some(other.width()) && rel_height >= Some(other.height())
        }
    }

    /// Bring the rectangle into normalized form where minimum and maximum form a true interval.
    #[must_use]
    pub fn normalize(self) -> Rectangle {
        Rectangle {
            x: self.x,
            y: self.y,
            max_x: self.x + self.width(),
            max_y: self.y + self.width(),
        }
    }

    /// A rectangle that the overlap of the two.
    #[must_use]
    pub fn meet(self, other: Self) -> Rectangle {
        Rectangle {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        }
    }

    /// A rectangle that contains both.
    #[must_use]
    pub fn join(self, other: Self) -> Rectangle {
        Rectangle {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Remove border from all sides.
    /// When the image is smaller than `border` in some dimension then the result is empty and
    /// contained in the original image but otherwise unspecified.
    #[must_use]
    pub fn inset(self, border: u32) -> Self {
        Rectangle {
            x: self.x.saturating_add(border),
            y: self.y.saturating_add(border),
            max_x: self.max_x.saturating_sub(border),
            max_y: self.max_y.saturating_sub(border),
        }
    }
}

impl CommandError {
    /// Indicates a very generic type error.
    const TYPE_ERR: Self = CommandError {
        type_err: true,
    };

    /// Indicates a very generic other error.
    /// E.g. the usage of a command requires an extension? Not quite sure yet.
    const OTHER: Self = CommandError {
        type_err: false,
    };

    pub fn is_type_err(&self) -> bool {
        self.type_err
    }
}
