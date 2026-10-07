//! Placeholder until the engine module lands.

use clap::ArgMatches;

use crate::common::*;

pub fn dispatch(_m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    Ok(ctx.fail("not yet available"))
}
