// Build script for static linking of minimal ONNX Runtime
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Enable model conversion feature by default or strictly for this build
    convert_models(&manifest_dir);

    // Only run this for the 'static' feature
    if !cfg!(feature = "static") {
        return;
    }

    let lib_dir = PathBuf::from(&manifest_dir)
        .join("onnxruntime-minimal")
        .join("lib");

    if !lib_dir.exists() {
        panic!(
            "Static ONNX Runtime libraries not found at {:?}. Run ./build-minimal-docker.sh first.",
            lib_dir
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link the unified libonnxruntime.a composed by build.sh
    // We use +whole-archive to ensure symbols like OrtGetApiBase are included
    // even if not directly referenced by Rust code (required for ort crate initialization)
    println!("cargo:rustc-link-lib=static:+whole-archive=onnxruntime");

    // Auto-discover and link Abseil libraries (libabsl_*.a)
    // We sort them to ensure deterministic build order, though cyclic deps might require groups.
    let mut absl_libs = Vec::new();
    if let Ok(entries) = fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap();
            if name.starts_with("libabsl_") && name.ends_with(".a") {
                let lib_name = &name[3..name.len() - 2]; // remove "lib" and ".a"
                absl_libs.push(lib_name.to_string());
            }
        }
    }
    absl_libs.sort();

    for lib in absl_libs {
        println!("cargo:rustc-link-lib=static={}", lib);
    }

    // Link other dependencies found in the dir
    // We check existence to be safe
    let deps = [
        "protobuf",
        "protobuf-lite",
        "nsync_cpp",
        "cpuinfo",
        "flatbuffers",
    ];
    for dep in deps {
        let filename = format!("lib{}.a", dep);
        if lib_dir.join(&filename).exists() {
            println!("cargo:rustc-link-lib=static={}", dep);
        }
    }

    // Link system libraries
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=stdc++"); // Important for C++ runtime
}

fn convert_models(manifest_dir: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Define models to process: (source_name, output_name)
    // Source path is relative to manifest_dir + ../stream/onnx_models
    let models = [
        ("gtcrn.onnx", "gtcrn.ort"),
        ("gtcrn_simple.onnx", "gtcrn_simple.ort"),
    ];

    let stream_dir = PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("stream")
        .join("onnx_models");
    let python_path = PathBuf::from(manifest_dir).join(".venv/bin/python");

    if !python_path.exists() {
        println!("cargo:warning=Python venv not found at {:?}, skipping model conversion. Ensure .venv exists if you need to convert models.", python_path);
        return;
    }

    for (src_name, dst_name) in models {
        let src_path = stream_dir.join(src_name);
        let dst_path = out_dir.join(dst_name);

        println!("cargo:rerun-if-changed={}", src_path.display());

        if !src_path.exists() {
            println!(
                "cargo:warning=Source model not found at {:?}, skipping conversion.",
                src_path
            );
            continue;
        }

        // Check if we need to convert (dst doesn't exist or src is newer)
        let should_convert = if !dst_path.exists() {
            true
        } else {
            let src_meta = fs::metadata(&src_path).unwrap();
            let dst_meta = fs::metadata(&dst_path).unwrap();
            src_meta.modified().unwrap() > dst_meta.modified().unwrap()
        };

        if should_convert {
            println!("Converting {} to ORT format...", src_name);
            let status = std::process::Command::new(&python_path)
                .args([
                    "-m",
                    "onnxruntime.tools.convert_onnx_models_to_ort",
                    src_path.to_str().unwrap(),
                    "--output_dir",
                    out_dir.to_str().unwrap(),
                    "--optimization_style",
                    "Fixed", // Use "Fixed" for pre-optimized models or "Runtime"
                             // Note: If using "Fixed", we might want to check the specific flags.
                             // For now, default behavior of the tool is usually sufficient.
                             // The tool output filename might default to model.ort, we need to handle renaming if needed,
                             // but usually it keeps the basename.
                ])
                .status()
                .expect("Failed to run conversion command");

            if !status.success() {
                panic!("Model conversion failed for {}", src_name);
            }

            // The tool might invoke "saved to <out_dir>/gtcrn.ort"
            // Ensure the expected output name matches
        }
    }
}
