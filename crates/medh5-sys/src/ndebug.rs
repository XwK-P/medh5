//! Whether the HDF5 this crate links was compiled with `NDEBUG`.
//!
//! cmake-rs, which builds HDF5 for `hdf5-metno-sys`, replaces CMake's release
//! flags under MSVC's Visual Studio generator --- `/O2 /Ob2 /DNDEBUG` --- with
//! the flags `cc` derives, so unless the environment adds `NDEBUG` an
//! optimised HDF5 keeps its `assert`s, and a damaged file aborts the process
//! where every other build returns the error the validator reports.  This
//! repository's `.cargo/config.toml` adds it; a build from crates.io never
//! reads that file.  The build script includes this module and stops such a
//! build rather than produce that binary (W02 of the 2.0 audit).
//!
//! It reads the flags HDF5 *was* compiled with --- the `CFLAGS` line of the
//! `libhdf5.settings` HDF5 installs --- not the environment: `hdf5-metno-src`
//! does not rebuild when `CFLAGS_<target>` changes, so an environment that now
//! carries `NDEBUG` says nothing about an HDF5 built before it did.

/// Set to build anyway: a build that aborts on a damaged file is acceptable,
/// or the settings file does not tell the truth.
pub const OPT_OUT: &str = "MEDH5_SYS_SKIP_NDEBUG_CHECK";

/// The flags line of a `libhdf5.settings`: CMake's flags for C and for the
/// build type, together --- what reached the compiler.
pub fn settings_cflags(settings: &str) -> Option<&str> {
    settings.lines().find_map(|line| line.trim_start().strip_prefix("CFLAGS:")).map(str::trim)
}

/// Whether `flags` define `NDEBUG`: `/DNDEBUG`, `-DNDEBUG`, `/D NDEBUG`, with
/// or without a value.
pub fn defines_ndebug(flags: &str) -> bool {
    let mut tokens = flags.split_ascii_whitespace();
    while let Some(token) = tokens.next() {
        let name = match token.strip_prefix("/D").or_else(|| token.strip_prefix("-D")) {
            Some("") => tokens.next().unwrap_or_default(),
            Some(name) => name,
            None => continue,
        };
        if name == "NDEBUG" || name.starts_with("NDEBUG=") {
            return true;
        }
    }
    false
}

/// Why the build must stop, or `None` when it may go ahead: an unoptimised
/// build (whose HDF5 keeps its assertions, as a debug build should), an HDF5
/// compiled with `NDEBUG`, or the opt-out set.  `settings` is the settings
/// file's location and text, or why it could not be read.
pub fn refusal(
    target: &str,
    msvc: bool,
    optimised: bool,
    settings: Result<(&str, &str), &str>,
    opt_out: Option<&str>,
) -> Option<String> {
    if !optimised || opt_out.is_some_and(|value| !value.is_empty() && value != "0") {
        return None;
    }
    let unknown = |why: &str| {
        Some(format!(
            "medh5-sys: cannot tell whether HDF5 was compiled with NDEBUG: {why}.\n\n\
             Set {OPT_OUT}=1 to build anyway."
        ))
    };
    let (place, text) = match settings {
        Ok(found) => found,
        Err(why) => return unknown(why),
    };
    let Some(flags) = settings_cflags(text) else {
        return unknown(&format!("{place} has no CFLAGS line"));
    };
    if defines_ndebug(flags) {
        return None;
    }
    let var = format!("CFLAGS_{}", target.replace(['-', '.'], "_"));
    let define = if msvc { "/DNDEBUG" } else { "-DNDEBUG" };
    Some(format!(
        "medh5-sys: the HDF5 this build links was compiled without NDEBUG, so it kept its \
         assertions: a damaged file would abort the process instead of returning an error.\n\n    \
         {place}\n    CFLAGS: {flags}\n\n\
         cmake-rs, which builds HDF5, replaces CMake's release flags under MSVC's Visual Studio \
         generator, /DNDEBUG included.  Give the C build NDEBUG in the environment:\n\n    \
         {var}={define}\n\n\
         (the medh5 repository's .cargo/config.toml does), then rebuild HDF5 with it --- it does \
         not rebuild when the variable changes: `cargo clean --release -p hdf5-metno-src` (a \
         `cargo install` starts clean).  Set {OPT_OUT}=1 to build anyway."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MSVC: &str = "x86_64-pc-windows-msvc";
    const PLACE: &str = "out/lib/libhdf5.settings";

    /// The lines of a real `libhdf5.settings` around the one read.
    fn settings(cflags: &str) -> String {
        format!(
            "                     Build Mode: Release\n\
             \x20                       Asserts: OFF\n\
             \x20                        CFLAGS: {cflags}\n\
             \x20                     H5_CFLAGS: /W3;/wd4100;-DNDEBUG\n\
             \x20                     AM_CFLAGS: \n"
        )
    }

    #[test]
    fn w02_ndebug_is_found_however_it_is_spelled() {
        for flags in ["/DNDEBUG", "-DNDEBUG", "/O2 /D NDEBUG", "-D NDEBUG=1", "/MD /DNDEBUG=1"] {
            assert!(defines_ndebug(flags), "{flags}");
        }
        for flags in ["", "/DNDEBUGX", "/D", "NDEBUG", "/UNDEBUG", "-DDEBUG"] {
            assert!(!defines_ndebug(flags), "{flags}");
        }
    }

    #[test]
    fn w02_the_flags_are_the_cflags_line_and_no_other() {
        let text = settings("/DWIN32 /D_WINDOWS /nologo /MD /Brepro");
        assert_eq!(settings_cflags(&text), Some("/DWIN32 /D_WINDOWS /nologo /MD /Brepro"));
        // H5_CFLAGS names NDEBUG here; it is not what reached the compiler.
        assert!(refusal(MSVC, true, true, Ok((PLACE, &text)), None).is_some());
        assert_eq!(settings_cflags("H5_CFLAGS: -DNDEBUG\n"), None);
    }

    #[test]
    fn w02_an_hdf5_compiled_without_ndebug_is_refused() {
        let text = settings("/DWIN32 /D_WINDOWS /nologo /MD /Brepro");
        let message = refusal(MSVC, true, true, Ok((PLACE, &text)), None).unwrap();
        for expected in [
            "CFLAGS_x86_64_pc_windows_msvc=/DNDEBUG",
            "cargo clean --release -p hdf5-metno-src",
            PLACE,
            "/nologo /MD /Brepro",
            OPT_OUT,
        ] {
            assert!(message.contains(expected), "{expected} missing from:\n{message}");
        }
        let linux = "x86_64-unknown-linux-gnu";
        let message = refusal(linux, false, true, Ok((PLACE, &settings("-O3"))), None).unwrap();
        assert!(message.contains("CFLAGS_x86_64_unknown_linux_gnu=-DNDEBUG"), "{message}");
    }

    #[test]
    fn w02_an_hdf5_compiled_with_it_builds() {
        for flags in ["/DWIN32 /nologo /MD /DNDEBUG", "-std=c11 -fPIC -O3 -DNDEBUG"] {
            assert!(refusal(MSVC, true, true, Ok((PLACE, &settings(flags))), None).is_none(), "{flags}");
        }
    }

    #[test]
    fn w02_what_cannot_be_read_is_refused_too() {
        let message = refusal(MSVC, true, true, Err("no such file"), None).unwrap();
        assert!(message.contains("no such file") && message.contains(OPT_OUT), "{message}");
        let message = refusal(MSVC, true, true, Ok((PLACE, "Build Mode: Release\n")), None).unwrap();
        assert!(message.contains("has no CFLAGS line"), "{message}");
    }

    #[test]
    fn w02_a_debug_build_and_the_opt_out_build() {
        let text = settings("/MD /Zi");
        // An unoptimised build keeps HDF5's assertions on every target.
        assert!(refusal(MSVC, true, false, Ok((PLACE, &text)), None).is_none());
        assert!(refusal(MSVC, true, false, Err("unread"), None).is_none());
        assert!(refusal(MSVC, true, true, Ok((PLACE, &text)), Some("1")).is_none());
        // An empty or zero opt-out is not one.
        assert!(refusal(MSVC, true, true, Ok((PLACE, &text)), Some("")).is_some());
        assert!(refusal(MSVC, true, true, Ok((PLACE, &text)), Some("0")).is_some());
    }
}
