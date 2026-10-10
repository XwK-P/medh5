fn main() {
    let code = medh5_cli::run_os(std::env::args_os().skip(1));
    std::process::exit(code);
}
