import datetime
import pathlib
import re
import urllib.error
import urllib.request


def _markdown_link(url: str, text: str, fallback: str) -> str:
    try:
        url_valid_code = 200
        url_valid = urllib.request.urlopen(url).getcode() == url_valid_code
    except (urllib.error.HTTPError, urllib.error.URLError):
        url_valid = False

    if not url_valid:
        print(f'Expanded link "{url}" unreachable. Skip modifying.')  # noqa: T201
        return fallback

    return f"[{text}]({url})"


def github_pr(match: re.Match[str]) -> str:
    link = _markdown_link(
        url=f"https://github.com/vc-bonn/torchhull/pull/{match.group(2)}",
        text=f"\\{match.group(1)}",
        fallback=f"{match.group(1)}",
    )
    return f"({link})"


def github_user(match: re.Match[str]) -> str:
    link = _markdown_link(
        url=f"https://github.com/{match.group(2)}",
        text=f"{match.group(1)}",
        fallback=f"{match.group(1)}",
    )
    return f" {link}"


def main() -> None:
    project_root_dir = pathlib.Path(__file__).parents[1]
    filename = project_root_dir / "CHANGELOG.md"

    with filename.open("r") as f:
        content = f.read()

    # Update links
    content = re.sub(pattern=R"\((#([0-9]+))\)", repl=github_pr, string=content)
    content = re.sub(pattern=R" (@([a-zA-Z0-9-]{0,37}[a-zA-Z0-9]))", repl=github_user, string=content)

    # Check date
    dates = re.findall(pattern=R"[0-9]{4}-[0-9]{2}-[0-9]{2}", string=content)
    timezone = datetime.timezone.utc
    today = datetime.datetime.now(timezone).date().isoformat()
    if not any(d == today for d in dates):
        msg = f"Did not find dates matching today.\n  Dates: {dates}\n  Today ({timezone}): {today}"
        raise ValueError(msg)

    with filename.open("w") as f:
        f.write(content)
        print(f'Updated "{filename}"')  # noqa: T201


if __name__ == "__main__":
    main()
