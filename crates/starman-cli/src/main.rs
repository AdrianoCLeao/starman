use std::process::ExitCode;

use clap::Parser;
use starman_cli::{dispatch, Cli};

fn main() -> ExitCode {
    let _ = env_logger::try_init();
    let cli = Cli::parse();

    match dispatch(cli) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            log::error!(target: "starman::cli", "{error}");
            eprintln!("error: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
