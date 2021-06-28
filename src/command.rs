use std::collections::HashMap;
use crate::buffer::{BufferLayout, Color, ColorChannel, Descriptor, Texel, Whitepoint};
use crate::program::{
    CompileError, Function, ImageBufferPlan, ImageBufferAssignment, PaintOnTopKind, Program,
    Texture,
};
use crate::pool::PoolImage;

/// A reference to one particular value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Register(pub(crate) usize);

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
#[derive(Default)]
pub struct CommandBuffer {
    ops: Vec<Op>,
}

#[derive(Clone)]
enum Op {
    /// i := in()
    Input {
        desc: Descriptor,
    },
    /// out(src)
    ///
    /// WIP: and is_cpu_type(desc)
    /// for the eventuality of gpu-only buffer layouts.
    Output {
        src: Register,
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
        desc: Descriptor,
    },
    /// i := binary(lhs, rhs)
    /// where type(i) =? Op[type(lhs), type(rhs)]
    Binary {
        lhs: Register,
        rhs: Register,
        op: BinaryOp,
        desc: Descriptor,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum ConstructOp {
    // TODO: can optimize this repr for the common case.
    Solid(Vec<u8>),
}

/// A high-level, device independent, translation of ops.
/// The main difference to Op is that this is no longer in SSA-form, and it may reinterpret and
/// reuse resources. In particular it will ran after the initial liveness analysis.
/// This will also return the _available_ strategies for one operation. For example, some texels
/// can not be represented on the GPU directly, depending on available formats, and need to be
/// either processed on the CPU (with SIMD hopefully) or they must be converted first, potentially
/// in a compute shader.
#[derive(Clone, Debug)]
pub(crate) enum High {
    /// Assign a texture id to an input with given descriptor.
    /// This instructs the program to insert instructions that load the image from the input in the
    /// pool into the associated texture buffer.
    Input(Register, Descriptor),
    /// Designate the ith textures as output n, according to the position in sequence of outputs.
    Output {
        /// The source register/texture/buffers.
        src: Register,
        /// The target texture.
        dst: Register,
    },
    #[deprecated = "Should be mapped to of paint with a discarding load or another buffer initialization."]
    Construct {
        dst: Texture,
        op: ConstructOp,
    },
    Paint {
        texture: Texture,
        dst: Target,
        fn_: Function,
    },
    /// Last phase marking a register as done.
    /// This is emitted after the Command defining the register has been translated.
    Done(usize),
}

/// The target image texture of a paint operation (pipeline).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    /// The data in the texture is to be discarded.
    Discard(Texture),
    /// The data in the texture must be loaded.
    Load(Texture),
}

#[derive(Clone)]
pub(crate) enum UnaryOp {
    /// Op = id
    Affine(Affine),
    /// Op = id
    Crop(Rectangle),
    /// Op(color)[T] = T[.color=color]
    /// And color needs to be 'color compatible' with the prior T (see module).
    ColorConvert(Color),
    /// Op(T) = T[.color=select(channel, color)]
    Extract { channel: ColorChannel },
}

#[derive(Clone)]
pub(crate) enum BinaryOp {
    /// Op[T, U] = T
    /// where T = U
    Inscribe { placement: Rectangle },
    /// Replace a channel T with U itself.
    /// Op[T, U] = T
    /// where select(channel, T.color) = U.color
    Inject { channel: ColorChannel }
}

/// A rectangle in `u32` space.
/// It's describe by minimum and maximum coordinates, inclusive and exclusive respectively. Any
/// rectangle where the order is not correct is interpreted as empty. This has the advantage of
/// simplifying certain operations that would otherwise need to check for correctness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rectangle {
    pub x: u32,
    pub y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Blend {
    Alpha,
}

#[derive(Clone, Copy)]
pub struct Affine {
    transformation: [f32; 9],
}

/// Reference of matrices and more: http://brucelindbloom.com/index.html?Eqn_ChromAdapt.html
///
/// A similar technique can simulate cone deficiencies:
/// * deuteranomaly (green cone cells defective),
/// * protanomaly (red cone cells defective),
/// * and tritanomaly (blue cone cells defective).
/// More information here: http://colorspace.r-forge.r-project.org/articles/color_vision_deficiency.html
///
/// Matrix for transforming cone response into the opponent color space which is assumed to be a
/// mostly sufficient input to recreate a particular color impression. In other words, simulate a
/// dichromacy response to a color by matching it for a 'standard' human. We can also estimate if a
/// particular color can be made visible for someone afflicted with a cone deficiency etc.
/// (see: Gustavo M. Machado, et.al A Physiologically-based Model for Simulation of Color Vision Deficiency)
/// 
/// ```text
/// 0.600    0.400    0.000
/// 0.240    0.105   −0.700
/// 1.200   −1.600    0.400
/// ```
pub struct ChromaticAdaptation {
    /// The target whitepoint of the adaptation.
    target: Whitepoint,
    /// The method to use.
    method: ChromaticAdaptationMethod,
}

pub enum ChromaticAdaptationMethod {
    /// Naive adaptation based on component-wise linear transform in XYZ.
    Xyz,
    /// A component-wise transform in LMS (cone response) coordinates.
    ///
    /// The matrix for whitepoint E (equal intensity) is:
    ///
    /// ```latex
    /// \begin{bmatrix}
    /// 0.38971 & 0.68898 & -0.07868\\
    /// -0.22981 & 1.18340 & 0.04641\\
    /// 0.00000 & 0.00000 & 1.00000
    /// \end{bmatrix}
    /// ```
    ///
    /// The D65 normalized XYZ -> LMS matrix is:
    ///
    /// ```text
    /// 0.4002400  0.7076000 -0.0808100
    /// -0.2263000  1.1653200  0.0457000
    /// 0.0000000  0.0000000  0.9182200
    /// ```
    VonKries,
    /// Bradford's modified (sharpened) LMS definition with linear VonKries adaptation.
    /// Used in ICC.
    ///
    /// ```latex
    /// \begin{bmatrix}
    /// 0.8951 & 0.2664 & -0.1614 \\
    /// -0.7502 & 1.7135 & 0.0367 \\
    /// 0.0389 & -0.0685 & 1.0296
    /// \end{bmatrix}
    /// ```
    BradfordVonKries,
    /// Bradford's originally intended adaptation.
    BradfordNonLinear,
}

#[derive(Debug)]
pub struct CommandError {
    inner: CommandErrorKind,
}

#[derive(Debug)]
enum CommandErrorKind {
    BadDescriptor(Descriptor),
    ConflictingTypes(Descriptor, Descriptor),
    GenericTypeError,
    Other,
}

impl CommandBuffer {
    /// Declare an input.
    ///
    /// Inputs MUST later be bound from the pool during launch.
    pub fn input(&mut self, desc: Descriptor) -> Result<Register, CommandError> {
        if !desc.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(desc),
            });
        }

        Ok(self.push(Op::Input { desc }))
    }

    /// Declare an image as input.
    ///
    /// Returns its register if the image has a valid descriptor, otherwise returns an error.
    pub fn input_from(&mut self, img: PoolImage)
        -> Register
    {
        let descriptor = img.descriptor();
        self.input(descriptor)
            .expect("Pool image descriptor should be valid")
    }

    /// Select a rectangular part of an image.
    pub fn crop(&mut self, src: Register, rect: Rectangle)
        -> Result<Register, CommandError>
    {
        let desc = self.describe_reg(src)?.clone();
        Ok(self.push(Op::Unary {
            src,
            op: UnaryOp::Crop(rect),
            desc,
        }))
    }

    /// Create an image with different color encoding.
    ///
    /// This goes through linear RGB, not ICC, and requires the two models to have same whitepoint.
    pub fn color_convert(&mut self, src: Register, texel: Texel)
        -> Result<Register, CommandError>
    {
        let desc_src = self.describe_reg(src)?;

        // Pretend that all colors with the same whitepoint will be mapped from encoded to
        // linear RGB when loading, and re-encoded in target format when storing them. This is
        // almost correct, but not all GPUs will support all texel kinds. In particular
        // some channel orders or bit-field channels are likely to be unsupported. In these
        // cases, we will later add some temporary conversion.
        match (&desc_src.texel.color, &texel.color) {
            (
                Color::Xyz { whitepoint: wp_src, .. },
                Color::Xyz { whitepoint: wp_dst, .. },
            ) if wp_src == wp_dst => {},
            _ => return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(desc_src.clone()),
            }),
        }

        // FIXME: validate memory condition.
        let layout = BufferLayout {
            width: desc_src.layout.width,
            height: desc_src.layout.height,
            // TODO: just add a bytes_u8 method or so.
            bytes_per_texel: texel.samples.bits.bytes() as u8,
            // TODO: make this nicer.
            bytes_per_row: desc_src.layout.width * texel.samples.bits.bytes() as u32,
        };

        let op = Op::Unary {
            src,
            op: UnaryOp::ColorConvert(texel.color.clone()),
            desc: Descriptor {
                layout,
                texel,
            },
        };

        Ok(self.push(op))
    }

    /// Perform a whitepoint adaptation.
    pub fn chromatic_adaptation(&mut self, _: Register, _: Whitepoint)
        -> Result<Register, CommandError>
    {
        todo!()
    }

    /// Embed this image as part of a larger one.
    pub fn inscribe(&mut self, below: Register, rect: Rectangle, above: Register)
        -> Result<Register, CommandError>
    {
        let desc_below = self.describe_reg(below)?;
        let desc_above = self.describe_reg(above)?;

        if desc_above.texel != desc_below.texel {
            return Err(CommandError {
                inner: CommandErrorKind::ConflictingTypes(desc_below.clone(), desc_above.clone()),
            });
        }

        if Rectangle::with_layout(&desc_above.layout) != rect {
            return Err(CommandError::OTHER);
        }

        if !Rectangle::with_layout(&desc_below.layout).contains(rect) {
            return Err(CommandError::OTHER);
        }

        let op = Op::Binary {
            lhs: below,
            rhs: above,
            op: BinaryOp::Inscribe {
                placement: rect.normalize(),
            },
            desc: desc_below.clone(),
        };

        Ok(self.push(op))
    }

    /// Extract some channels from an image data into a new view.
    pub fn extract(&mut self, src: Register, channel: ColorChannel)
        -> Result<Register, CommandError>
    {
        let desc = self.describe_reg(src)?;
        let texel = desc.channel_texel(channel)
            .ok_or_else(|| CommandError::OTHER)?;
        let op = Op::Unary {
            src,
            op: UnaryOp::Extract { channel },
            desc: Descriptor {
                layout: desc.layout.clone(),
                texel,
            },
        };

        Ok(self.push(op))
    }

    /// Overwrite some channels with overlaid data.
    pub fn inject(&mut self, below: Register, channel: ColorChannel, above: Register)
        -> Result<Register, CommandError>
    {
        let desc_below = self.describe_reg(below)?;
        let expected_texel = desc_below.channel_texel(channel)
            .ok_or_else(|| CommandError::OTHER)?;
        let desc_above = self.describe_reg(above)?;

        if expected_texel != desc_above.texel {
            return Err(CommandError {
                inner: CommandErrorKind::ConflictingTypes(desc_below.clone(), desc_above.clone()),
            });
        }

        let op = Op::Binary {
            lhs: below,
            rhs: above,
            op: BinaryOp::Inject { channel },
            desc: desc_below.clone(),
        };

        Ok(self.push(op))
    }

    /// Overlay this image as part of a larger one, performing blending.
    pub fn blend(&mut self, _below: Register, _rect: Rectangle, _above: Register, _blend: Blend)
        -> Result<Register, CommandError>
    {
        // TODO: What blending should we support
        Err(CommandError::OTHER)
    }

    /// A solid color image, from a descriptor and a single texel.
    pub fn solid(&mut self, describe: Descriptor, data: &[u8])
        -> Result<Register, CommandError>
    {
        if !describe.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        if data.len() != usize::from(describe.layout.bytes_per_texel) {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        Ok(self.push(Op::Construct {
            desc: describe,
            op: ConstructOp::Solid(data.to_owned()),
        }))
    }

    /// An affine transformation of the image.
    pub fn affine(&mut self, src: Register, affine: Affine)
        -> Result<Register, CommandError>
    {
        // TODO: should we check affine here?
        let desc = self.describe_reg(src)?.clone();
        Ok(self.push(Op::Unary {
            src,
            op: UnaryOp::Affine(affine),
            desc,
        }))
    }

    /// Declare an output.
    ///
    /// Outputs MUST later be bound from the pool during launch.
    pub fn output(&mut self, src: Register)
        -> Result<(Register, Descriptor), CommandError>
    {
        let outformat = self.describe_reg(src)?.clone();
        // Ignore this, it doesn't really produce a register.
        let register = self.push(Op::Output {
            src,
        });
        Ok((register, outformat))
    }

    pub fn compile(&self) -> Result<Program, CompileError> {
        let steps = self.ops.len();

        let mut last_use = vec![0; steps];
        let mut first_use = vec![steps; steps];

        let mut high_ops = vec![];

        // Liveness analysis.
        for (back_idx, op) in self.ops.iter().rev().enumerate() {
            let idx = self.ops.len() - 1 - back_idx;
            match op {
                Op::Input { .. } | Op::Construct { .. } => {},
                &Op::Output { src: Register(src) } => {
                    last_use[src] = last_use[src].max(idx);
                    first_use[src] = first_use[src].min(idx);
                },
                &Op::Unary { src: Register(src), .. } => {
                    last_use[src] = last_use[src].max(idx);
                    first_use[src] = first_use[src].min(idx);
                },
                &Op::Binary { lhs: Register(lhs), rhs: Register(rhs), .. } => {
                    last_use[rhs] = last_use[rhs].max(idx);
                    first_use[rhs] = first_use[rhs].min(idx);
                    last_use[lhs] = last_use[lhs].max(idx);
                    first_use[lhs] = first_use[lhs].min(idx);
                },
            }
        }

        let mut textures = ImageBufferPlan::default();
        let mut reg_to_texture: HashMap<Register, Texture> = HashMap::default();

        for (idx, op) in self.ops.iter().enumerate() {
            let liveness = first_use[idx]..last_use[idx];
            let descriptor = self.describe_reg(if let &Op::Output { src } = op {
                src
            } else {
                Register(idx)
            }).expect("A non-output register");

            let ImageBufferAssignment { buffer, texture }
                = textures.allocate_for(descriptor, liveness);

            match op {
                Op::Input { desc } => {
                    high_ops.push(High::Input(Register(idx), desc.clone()));
                    reg_to_texture.insert(Register(idx), texture);
                }
                &Op::Output { src } => {
                    high_ops.push(High::Output {
                        src,
                        dst: Register(idx),
                    });
                }
                Op::Construct { desc: _, op } => {
                    high_ops.push(High::Construct {
                        dst: texture,
                        op: op.clone(),
                    });
                    reg_to_texture.insert(Register(idx), texture);
                }
                Op::Unary { desc: _, src, op } => {
                    match op {
                        &UnaryOp::Crop(region) => {
                            let target = Rectangle::with_width_height(region.width(), region.height());
                            high_ops.push(High::Paint {
                                texture: reg_to_texture[src],
                                dst: Target::Discard(texture),
                                fn_: Function::PaintOnTop {
                                    selection: region,
                                    target,
                                    viewport: target,
                                    paint_on_top: PaintOnTopKind::Copy,
                                },
                            });
                        },
                        _ => return Err(CompileError::NotYetImplemented),
                    }

                    reg_to_texture.insert(Register(idx), texture);
                }
                Op::Binary { desc: _, lhs, rhs, op } => {
                    let lower_region = Rectangle::from(self.describe_reg(*lhs).unwrap());
                    let upper_region = Rectangle::from(self.describe_reg(*rhs).unwrap());

                    match op {
                        BinaryOp::Inscribe { placement } => {
                            high_ops.push(High::Paint {
                                dst: Target::Discard(texture),
                                texture: reg_to_texture[lhs],
                                fn_: Function::PaintOnTop {
                                    selection: lower_region,
                                    target: lower_region,
                                    viewport: lower_region,
                                    paint_on_top: PaintOnTopKind::Copy,
                                },
                            });

                            high_ops.push(High::Paint {
                                dst: Target::Load(texture),
                                texture: reg_to_texture[rhs],
                                fn_: Function::PaintOnTop {
                                    selection: upper_region,
                                    target: *placement,
                                    viewport: lower_region,
                                    paint_on_top: PaintOnTopKind::Copy,
                                },
                            });
                        },
                        _ => return Err(CompileError::NotYetImplemented),
                    }

                    reg_to_texture.insert(Register(idx), texture);
                }
            }

            high_ops.push(High::Done(idx));
        }

        Ok(Program {
            ops: high_ops,
            textures,
        })
    }

    /// Get the descriptor for a register.
    fn describe_reg(&self, Register(reg): Register)
        -> Result<&Descriptor, CommandError>
    {
        match self.ops.get(reg) {
            None | Some(Op::Output { .. }) => {
                Err(CommandError::BAD_REGISTER)
            }
            Some(Op::Input { desc })
            | Some(Op::Construct { desc, .. })
            | Some(Op::Unary { desc, .. })
            | Some(Op::Binary { desc, .. }) => {
                Ok(desc)
            }
        }
    }

    fn push(&mut self, op: Op) -> Register {
        let reg = Register(self.ops.len());
        self.ops.push(op);
        reg
    }
}

impl Rectangle {
    /// A rectangle at the origin with given width (x) and height (y).
    pub fn with_width_height(width: u32, height: u32) -> Self {
        Rectangle { x: 0, y: 0, max_x: width, max_y: height }
    }

    /// A rectangle describing a complete buffer.
    pub fn with_layout(buffer: &BufferLayout) -> Self {
        Self::with_width_height(buffer.width, buffer.height)
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

    /// The meet, relative to the coordinates of this rectangle.
    #[must_use]
    pub fn meet_in_local_coordinates(self, other: Self) -> Rectangle {
        // Normalize to ensure that max_{x,y} is not less than {x,y}
        let meet = self.normalize().meet(other);
        Rectangle {
            x: meet.x - self.x,
            y: meet.y - self.y,
            max_x: meet.max_x - self.x,
            max_y: meet.max_y - self.y,
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

impl From<&'_ BufferLayout> for Rectangle {
    fn from(buffer: &BufferLayout) -> Rectangle {
        Rectangle::with_width_height(buffer.width, buffer.height)
    }
}

impl From<&'_ Descriptor> for Rectangle {
    fn from(buffer: &Descriptor) -> Rectangle {
        Rectangle::from(&buffer.layout)
    }
}

impl CommandError {
    /// Indicates a very generic type error.
    const TYPE_ERR: Self = CommandError {
        inner: CommandErrorKind::GenericTypeError,
    };

    /// Indicates a very generic other error.
    /// E.g. the usage of a command requires an extension? Not quite sure yet.
    const OTHER: Self = CommandError {
        inner: CommandErrorKind::Other,
    };

    /// Specifies that a register reference was invalid.
    const BAD_REGISTER: Self = Self::OTHER;

    pub fn is_type_err(&self) -> bool {
        matches!(self.inner,
            CommandErrorKind::GenericTypeError
            | CommandErrorKind::ConflictingTypes(_, _)
            | CommandErrorKind::BadDescriptor(_))
    }
}

#[test]
fn rectangles() {
    let small = Rectangle::with_width_height(2, 2);
    let large = Rectangle::with_width_height(4, 4);

    assert_eq!(large, large.join(small));
    assert_eq!(small, large.meet(small));
    assert!(large.contains(small));
    assert!(!small.contains(large));
}

#[test]
fn simple_program() {
    use crate::pool::Pool;
    use image::GenericImageView;

    const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
    const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

    let mut pool = Pool::new();
    let mut commands = CommandBuffer::default();

    let background = image::open(BACKGROUND)
        .expect("Background image opened");
    let foreground = image::open(FOREGROUND)
        .expect("Background image opened");
    let expected = BufferLayout::from(&background);

    let placement = Rectangle {
        x: 0,
        y: 0,
        max_x: foreground.width(),
        max_y: foreground.height(),
    };

    let background = pool.insert_srgb(&background);
    let background = commands.input_from(background.into());

    let foreground = pool.insert_srgb(&foreground);
    let foreground = commands.input_from(foreground.into());

    let result = commands.inscribe(background, placement, foreground)
        .expect("Valid to inscribe");
    let (_, outformat) = commands.output(result)
        .expect("Valid for output");

    let _ = commands.compile()
        .expect("Could build command buffer");
    assert_eq!(outformat.layout, expected);
}
