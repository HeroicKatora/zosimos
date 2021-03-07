use std::{env, fs, io, io::Read as _, path};
use shaderc::{Compiler, ShaderKind};

#[derive(Debug)]
enum BuildError {
    Io(io::Error),
    Shader(shaderc::Error),
}

fn main() -> Result<(), BuildError> {
    struct SimpleSource {
        path: &'static str,
        kind: ShaderKind,
    }

    const SHADERS: &[SimpleSource] = &[
        SimpleSource {
            path: "src/shaders/noop.vert",
            kind: ShaderKind::Vertex,
        },
        SimpleSource {
            path: "src/shaders/copy.frag",
            kind: ShaderKind::Fragment,
        },
        SimpleSource {
            path: "src/shaders/inject.frag",
            kind: ShaderKind::Fragment,
        },
    ];

    let mut compiler = Compiler::new().unwrap();
    let mut str_source = String::new();

    let target_dir = env::var_os("OUT_DIR").unwrap();
    let target_dir = path::Path::new(&target_dir).join("spirv");

    fs::create_dir_all(&target_dir)?;

    for shader in SHADERS {
        println!("cargo:rerun-if-changed={}", shader.path);
        let mut file = fs::File::open(shader.path)?;
        str_source.clear();
        file.read_to_string(&mut str_source)?;
        
        let binary = compiler.compile_into_spirv(
            &str_source,
            shader.kind, 
            shader.path,
            "main",
            None)?;

        let spirv = binary.as_binary_u8();

        let mut path = target_dir.clone();
        let filepath = path::Path::new(shader.path);
        assert!(filepath.is_relative());

        path.push(filepath.file_name().unwrap());
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
