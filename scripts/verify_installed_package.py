from __future__ import annotations

from importlib.metadata import entry_points, version
from importlib.resources import files


REQUIRED_PROMPTS = (
    "element_expansion.md",
    "idea_divergence.md",
    "idea_mixer.md",
    "idea_similarity.md",
    "script_continuity_reviewer.md",
    "script_rewriter.md",
    "script_writer.md",
    "selfplay_creator.md",
    "story_continuity_reviewer.md",
    "story_rewriter.md",
    "story_writer.md",
)


def main() -> None:
    package_root = files("guided_story_agent")
    required_resources = tuple(
        package_root.joinpath("prompts", name) for name in REQUIRED_PROMPTS
    ) + (package_root.joinpath("resources", "batch_cases.jsonl"),)
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError("安装包缺少资源：" + "；".join(missing))

    scripts = {
        item.name
        for item in entry_points(group="console_scripts")
        if item.value.startswith("guided_story_agent.")
    }
    required_scripts = {
        "guided-story-web",
        "guided-story-cli",
        "guided-story-selfplay",
        "guided-story-batch",
    }
    missing_scripts = sorted(required_scripts - scripts)
    if missing_scripts:
        raise RuntimeError("安装包缺少命令入口：" + "、".join(missing_scripts))

    print(
        f"guided-story-video-agent {version('guided-story-video-agent')} package resources verified"
    )


if __name__ == "__main__":
    main()
