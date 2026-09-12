# RTSP / DeepStream Video Pipeline Experiments

This repository is a working area for NVIDIA DeepStream-based multi-stream video processing and storage/streaming experiments. It includes DeepStream sample applications together with local integration and benchmarking material used while developing RTSP-oriented video pipelines.

## Scope

The repository contains examples covering multi-stream inference, tracking, segmentation, analytics metadata, runtime source management, RTSP input/output, image-buffer access, and Triton-related DeepStream paths.

These examples are useful for testing the lower-level behavior required by larger multi-camera systems, including source multiplexing, GPU buffer access, inference metadata, and downstream stream handling.

## Upstream Attribution

A significant part of the `apps/` and `bindings/` material originates from NVIDIA DeepStream Python sample applications. Those components remain subject to their original authorship and licensing. This repository should be understood as a development/experimentation workspace built around those samples, not as an original implementation of the DeepStream SDK examples.

## Local Experiments

Local Docker commands, benchmark outputs, diagrams, and integration code are retained to document system-level experimentation around multi-camera video processing.

## Related Project

The more structured multi-camera application built from this experimentation is maintained in [deepstream_mediamtx](https://github.com/shosseinj/deepstream_mediamtx).


## Goal

The repository is a systems laboratory for testing DeepStream source handling, metadata access, RTSP output, and storage integration before those techniques are incorporated into a larger camera platform.

## Installation

A practical setup requires Linux, an NVIDIA GPU and driver, Docker with GPU support, and a DeepStream image compatible with the checked-in samples. Review the bind mounts and ports in the Compose files, then start the required stack with:

```bash
docker compose -f compose.rtsp.yaml up --build
```

Different Compose files represent different experiments; they are not interchangeable deployment profiles.

## Working with the Repository

Upstream DeepStream samples are under `apps/` and bindings under `bindings/`; local integration and benchmark files sit alongside them. Work from a copy of the relevant sample, keep camera URLs and object-store credentials in untracked configuration, and avoid treating checked-in benchmark JSON as performance evidence for different hardware. The tracked event and face-embedding databases require a privacy review before use.
