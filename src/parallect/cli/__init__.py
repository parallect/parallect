"""Typer CLI application for parallect — the research engine."""

import typer

from parallect.cli.config import config_cmd
from parallect.cli.continue_ import continue_cmd
from parallect.cli.enhance import enhance_cmd
from parallect.cli.research import research_cmd

app = typer.Typer(
    name="parallect",
    help="Multi-provider AI deep research from the command line.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.command("research")(research_cmd)
app.command("continue")(continue_cmd)
app.command("enhance")(enhance_cmd)
app.command("config")(config_cmd)


if __name__ == "__main__":
    app()
