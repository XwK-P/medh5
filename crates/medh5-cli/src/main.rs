fn main() {
    let code = medh5_cli::run(std::env::args().skip(1).collect());
    std::process::exit(code);
}
