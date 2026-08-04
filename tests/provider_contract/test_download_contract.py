from __future__ import annotations

from guided_story_agent.v2 import DownloadDestination


def test_download_uses_runtime_owned_part_and_final_paths(provider_and_transport, request_context, video_job, artifact_dir) -> None:
    provider, _ = provider_and_transport
    job = provider.submit(video_job, request_context).provider_job
    if provider.provider_key == "mock-http":
        provider.poll(job, request_context)
        provider.poll(job, request_context)
        job = provider.poll(job, request_context).provider_job
    destination = DownloadDestination(str(artifact_dir / "artifact.bin.part"), str(artifact_dir / "artifact.bin"))
    result = provider.download(job, destination, request_context)
    assert result.completed
    assert result.final_candidate_path == str(artifact_dir / "artifact.bin")
    assert (artifact_dir / "artifact.bin").exists()
