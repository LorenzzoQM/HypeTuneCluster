import logging

# Code copied/adapted from https://github.com/AVSLab/bsk_rl/

fstr = "\x1b[30;3m%(asctime)s\x1b[0m %(shortname)-30s %(levelname)-10s <%(sim_time)-.2f> %(message)s"

colors = dict(
    GRAY=90,
    RED=91,
    GREEN=92,
    YELLOW=93,
    BLUE=94,
    MAGENTA=95,
    CYAN=96,
    WHITE=97,
    DARK_GRAY=30,
    DARK_RED=31,
    DARK_GREEN=32,
    DARK_YELLOW=33,
    DARK_BLUE=34,
    DARK_MAGENTA=35,
    DARK_CYAN=36,
    DARK_WHITE=37,
)

level_color = {
    "DEBUG": None,
    "INFO": None,
    "WARNING": "YELLOW",
    "ERROR": "RED",
    "CRITICAL": "RED",
}


def style_string(
    string,
    no_format=False,
    style_spec=None,
    color=None,
    background_color=None,
    bold=False,
    emph=False,
    underline=False,
):
    if no_format:
        return string

    if style_spec is None:
        style_spec = []
    if color is not None:
        style_spec.append(colors[color.upper()])
    if background_color is not None:
        style_spec.append(colors[background_color.upper()] + 10)
    if bold:
        style_spec.append(1)
    if emph:
        style_spec.append(3)
    if underline:
        style_spec.append(4)
    return (
        "\x1b["
        + ";".join([str(style) for style in style_spec])
        + "m"
        + string
        + "\x1b[0m"
    )


class SimFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

        self.satellite_colors = {}

    def set_format(self, fmt, defaults=None, style="%"):
        self._style = logging._STYLES[style][0](fmt, defaults=defaults)
        self._fmt = self._style._fmt

    def format(self, record):
        fstr = ""
        fstr += style_string("%(asctime)s ", color="GRAY", emph=True)

        record.shortname = ".".join(record.name.split(".")[1:])
        fstr += style_string("%(shortname)-30s ", color="GRAY")
        fstr += style_string(
            "%(levelname)-10s ",
            color=level_color[record.levelname],
            bold=record.levelname == "CRITICAL",
        )
        if hasattr(record, "iteration"):
            fstr += style_string(
                "Iteration %(iteration)-.2f:",
                bold=True,
                color="GREEN",
            )
            fstr += " "

        fstr += style_string(
            "%(message)s",
            color=level_color[record.levelname],
            bold=record.levelname == "CRITICAL",
        )
        if hasattr(record, "time"):
            fstr += style_string(
                " %(time).2f seconds",
                color="DARK_CYAN",
            )

        self.set_format(fstr)
        return super().format(record)


def setup_logger(logger_level=logging.INFO):
    logger = logging.getLogger("hypetune")
    logger.setLevel(logger_level)
    ch = logging.StreamHandler()
    ch.setFormatter(SimFormatter())
    logger.addHandler(ch)
