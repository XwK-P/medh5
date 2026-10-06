fn main() {
    let code = medh5_cli::run(std::env::args().collect());
    std::process::exit(code);
}
