from __future__ import annotations

import unittest
from importlib.resources import files
from pathlib import Path

from guided_story_agent.agent import OpenAIStoryAgent
from guided_story_agent.v2.openai_director import OpenAIDirectorAgent


class PackageResourceTests(unittest.TestCase):
    def test_prompts_are_available_inside_the_package(self) -> None:
        prompt_root = files("guided_story_agent").joinpath("prompts")
        required = (
            "element_expansion.md",
            "idea_divergence.md",
            "idea_mixer.md",
            "idea_similarity.md",
            "quality_judge.md",
            "script_compressor.md",
            "script_continuity_reviewer.md",
            "script_rewriter.md",
            "story_writer.md",
            "script_writer.md",
            "selfplay_creator.md",
            "story_continuity_reviewer.md",
            "story_rewriter.md",
            "storyboard_director.md",
        )
        for name in required:
            with self.subTest(name=name):
                text = prompt_root.joinpath(name).read_text(encoding="utf-8")
                self.assertGreater(len(text.strip()), 20)

        agent = OpenAIStoryAgent(None, "offline-test")
        self.assertEqual(
            Path(str(prompt_root)).resolve(),
            Path(agent.prompt_dir).resolve(),
        )
        self.assertIn("故事创作者", agent._load_prompt("story_writer.md"))
        v2_prompt = prompt_root.joinpath("v2", "director", "movie_plan.md")
        self.assertGreater(len(v2_prompt.read_text(encoding="utf-8").strip()), 100)
        v2_agent = OpenAIDirectorAgent(None, "offline-test")
        self.assertEqual(v2_prompt.resolve(), Path(v2_agent.prompt_path).resolve())

    def test_default_batch_cases_are_packaged(self) -> None:
        source = files("guided_story_agent").joinpath("resources", "batch_cases.jsonl")
        lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(12, len(lines))


if __name__ == "__main__":
    unittest.main()
