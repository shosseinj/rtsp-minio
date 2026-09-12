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
