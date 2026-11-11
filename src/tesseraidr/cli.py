"""Command-line entry point: ``tesseraidr <command> [options]``."""

import sys

from . import __version__, DATA_DIR

COMMANDS = {
    "multiblock": ("tesseraidr.multiblock", "Sliding-window / multi-block GIN assignment."),
    "fullseq":    ("tesseraidr.fullseq",    "Full-sequence GIN assignment."),
    "summary":    ("tesseraidr.summary",    "out_summary -> block table, domain table, plots."),
    "data":       (None,                    "Print where the bundled reference files are."),
}

USAGE = "usage: tesseraidr {" + ",".join(COMMANDS) + "} [options]\n"


def _help() -> str:
    lines = [USAGE, f"TESSERA-IDR {__version__}\n", "commands:"]
    lines += [f"  {name:<11} {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run 'tesseraidr <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_help())
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"tesseraidr {__version__}")
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        sys.stderr.write(USAGE + f"tesseraidr: error: unknown command {cmd!r}\n")
        return 2

    if cmd == "data":
        print(DATA_DIR)
        for p in sorted(DATA_DIR.iterdir()):
            if p.is_file():
                print(f"  {p.name}")
        return 0

    from importlib import import_module
    module = import_module(COMMANDS[cmd][0])
    module.main(rest, prog=f"tesseraidr {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
