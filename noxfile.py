import pathlib
import shutil

import nox

nox.options.sessions = []


@nox.session
def format(session: nox.Session) -> None:  # noqa: A001
    """Format all source files to a consistent style."""
    sources = ["src", "tests", "docs", "benchmarks", "tools", "noxfile.py"]
    session.run("isort", *sources, external=True)
    session.run(
        "docformatter",
        "--in-place",
        "--recursive",
        *sources,
        external=True,
    )
    session.run("black", *sources, external=True)

    sources_cpp = ["src"]
    sources_cpp_files = []
    for s in sources_cpp:
        file = pathlib.Path(s)
        if file.is_file():
            sources_cpp_files.append(str(file))
        elif file.is_dir():
            for ext in [".h", ".hpp", ".cuh", ".c", ".cpp", ".cu"]:
                sources_cpp_files.extend([str(f) for f in sorted(file.rglob(f"*{ext}"))])
    session.run("clang-format", "-i", *sources_cpp_files, external=True)


DEFAULT_DOCS_OUTPUT_TYPE = "html"  # Replace by "dirhtml" for nice website URLs


@nox.session
def docs(session: nox.Session, output_type: str = DEFAULT_DOCS_OUTPUT_TYPE) -> None:
    """Build the documentation locally in the 'docs/build' directory."""
    # Cleanup
    shutil.rmtree("docs/_autosummary", ignore_errors=True)
    shutil.rmtree("build/docs/html", ignore_errors=True)

    # sphinx-build
    force_building = "-E"
    session.run(
        "sphinx-build",
        "-b",
        output_type,
        force_building,
        "-j",
        "auto",
        "docs",
        "build/docs/html",
        external=True,
    )

    # Hints
    session.debug("\n\n--> Open 'build/docs/html/index.html' to view the documentation\n")


@nox.session
def docs_live(session: nox.Session) -> None:
    """Build the documentation locally and starts an interactive live session with automatic rebuilding on changes."""
    output_type = "html"  # Do not use "dirhtml" since auto-refresh will not work properly

    # Perform a complete clean build before starting to watch changes to avoid infinite loop
    docs(session, output_type=output_type)

    # sphinx-build
    force_building = "-E"
    watch_dirs_and_files = ["src", "README.md", "CHANGELOG.md"]
    watch_args = [v for pair in zip(["--watch"] * len(watch_dirs_and_files), watch_dirs_and_files) for v in pair]
    session.run(
        "sphinx-autobuild",
        "-b",
        output_type,
        force_building,
        "-j",
        "auto",
        "--open-browser",
        *watch_args,
        "--re-ignore",
        "docs/_autosummary",
        "docs",
        "build/docs/html",
        external=True,
    )


WITH_MYPY = False


@nox.session
def lint(session: nox.Session) -> None:
    """Check the source code with linters."""
    failed = False
    sources = ["src", "tests", "docs", "benchmarks", "tools", "noxfile.py"]
    try:
        session.run("isort", "--check", *sources, external=True)
    except nox.command.CommandFailed:
        failed = True

    try:
        session.run(
            "docformatter",
            "--check",
            "--recursive",
            *sources,
            external=True,
        )
    except nox.command.CommandFailed:
        failed = True

    try:
        session.run("black", "--check", *sources, external=True)
    except nox.command.CommandFailed:
        failed = True

    try:
        sources_cpp = ["src"]
        sources_cpp_files = []
        for s in sources_cpp:
            file = pathlib.Path(s)
            if file.is_file():
                sources_cpp_files.append(str(file))
            elif file.is_dir():
                for ext in [".h", ".hpp", ".cuh", ".c", ".cpp", ".cu"]:
                    sources_cpp_files.extend([str(f) for f in sorted(file.rglob(f"*{ext}"))])
        session.run("clang-format", "--dry-run", "--Werror", *sources_cpp_files, external=True)
    except nox.command.CommandFailed:
        failed = True

    try:
        session.run("ruff", "check", *sources, external=True)
    except nox.command.CommandFailed:
        failed = True

    if WITH_MYPY:
        try:
            session.run("mypy", *sources, external=True)
        except nox.command.CommandFailed:
            failed = True

    try:
        text_sources = [*sources, "README.md", "CHANGELOG.md"]
        skip_sources = ["*pdf"]
        session.run(
            "codespell",
            "--check-filenames",
            "--check-hidden",
            *text_sources,
            "--skip",
            ",".join(skip_sources),
            external=True,
        )
    except nox.command.CommandFailed:
        failed = True

    try:
        session.run(
            "check-manifest",
            external=True,
        )
    except nox.command.CommandFailed:
        failed = True

    if failed:
        raise nox.command.CommandFailed


@nox.session
def benchmarks(session: nox.Session) -> None:
    """Runs the benchmarks."""
    session.run("pytest", "benchmarks", "--benchmark-sort=fullname", external=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.run("pytest", "tests", external=True)
