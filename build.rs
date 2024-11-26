use shaderc::{Compiler, ShaderKind};
use std::{env, fs, io, io::Read as _, path};

#[derive(Debug)]
#[allow(unused)]
enum BuildError {
    Io(io::Error),
    Shader(shaderc::Error),
}

fn main() -> Result<(), BuildError> {
    struct SimpleSource {
        path: &'static str,
        kind: ShaderKind,
        entry: &'static str,
        name_overwrite: Option<&'static str>,
    }

    const SHADERS: &[SimpleSource] = &[
        SimpleSource {
            path: "src/shaders/box.vert",
            kind: ShaderKind::Vertex,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/copy.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/inject.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/linear.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/stage.frag",
            kind: ShaderKind::Fragment,
            entry: "decode_r8ui",
            name_overwrite: Some("stage_d8ui"),
        },
        SimpleSource {
            path: "src/shaders/stage.frag",
            kind: ShaderKind::Fragment,
            entry: "encode_r8ui",
            name_overwrite: Some("stage_e8ui"),
        },
        SimpleSource {
            path: "src/shaders/stage.frag",
            kind: ShaderKind::Fragment,
            entry: "decode_r16ui",
            name_overwrite: Some("stage_d16ui"),
        },
        SimpleSource {
            path: "src/shaders/stage.frag",
            kind: ShaderKind::Fragment,
            entry: "encode_r16ui",
            name_overwrite: Some("stage_e16ui"),
        },
        SimpleSource {
            path: "src/shaders/stage.frag",
            kind: ShaderKind::Fragment,
            entry: "decode_r32ui",
            name_overwrite: Some("stage_d32ui"),
        },
        SimpleSource {
            path: "src/shaders/stage.frag",
            kind: ShaderKind::Fragment,
            entry: "encode_r32ui",
            name_overwrite: Some("stage_e32ui"),
        },
        SimpleSource {
            path: "src/shaders/fill.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/distribution_normal2d.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/fractal_noise.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/palette.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/bilinear.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/oklab.frag",
            kind: ShaderKind::Fragment,
            entry: "oklab_encode",
            name_overwrite: Some("oklab_encode"),
        },
        SimpleSource {
            path: "src/shaders/oklab.frag",
            kind: ShaderKind::Fragment,
            entry: "oklab_decode",
            name_overwrite: Some("oklab_decode"),
        },
        SimpleSource {
            path: "src/shaders/srlab2.frag",
            kind: ShaderKind::Fragment,
            entry: "srlab2_encode",
            name_overwrite: Some("srlab2_encode"),
        },
        SimpleSource {
            path: "src/shaders/srlab2.frag",
            kind: ShaderKind::Fragment,
            entry: "srlab2_decode",
            name_overwrite: Some("srlab2_decode"),
        },
        SimpleSource {
            path: "src/shaders/box3.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/mandelbrot.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/solid_rgb.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/crt.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
        SimpleSource {
            path: "src/shaders/flat_field.frag",
            kind: ShaderKind::Fragment,
            entry: "main",
            name_overwrite: None,
        },
    ];

    let compiler = Compiler::new().unwrap();
    let mut str_source = String::new();

    let target_dir = env::var_os("OUT_DIR").unwrap();
    let target_dir = path::Path::new(&target_dir).join("spirv");

    fs::create_dir_all(&target_dir)?;

    for shader in SHADERS {
        println!("cargo:rerun-if-changed={}", shader.path);
        let mut file = fs::File::open(shader.path)?;
        str_source.clear();
        file.read_to_string(&mut str_source)?;

        let mut options = shaderc::CompileOptions::new().expect("Could initialize compile options");
        if shader.entry != "main" {
            let macro_name = shader.entry.to_uppercase() + "_AS_MAIN";
            options.add_macro_definition(&macro_name, Some("main"));
        }

        let binary = compiler.compile_into_spirv(
            &str_source,
            shader.kind,
            shader.path,
            shader.entry,
            // Needed, but empty. Without this the `shader.entry` parameter is ignored!
            Some(&options),
        )?;

        let spirv = binary.as_binary_u8();

        let mut path = target_dir.clone();
        let filepath = path::Path::new(shader.path);
        assert!(filepath.is_relative());

        path.push(filepath.file_name().unwrap());
        if let Some(name) = shader.name_overwrite {
            path.set_file_name(name);
        }
        path.set_extension(match shader.kind {
            ShaderKind::Vertex => "vert.v",
            ShaderKind::Fragment => "frag.v",
            _ => unreachable!(),
        });

        fs::write(path, spirv)?;
    }

    Ok(())
}

impl From<io::Error> for BuildError {
    fn from(err: io::Error) -> Self {
        BuildError::Io(err)
    }
}

impl From<shaderc::Error> for BuildError {
    fn from(err: shaderc::Error) -> Self {
        BuildError::Shader(err)
    }
}
