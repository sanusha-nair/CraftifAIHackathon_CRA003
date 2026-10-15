 Restricted-Zone Intrusion Sentinel

 Objective
A computer-vision security project designed to detect people in video and support restricted-area monitoring.

 Tools and Technologies
- CraftifAI PipeGen
- NVIDIA DeepStream
- ONNX, if used in the generated pipeline

 Input
- Video file: `Samplevdo.mp4`
- Video path: `/home/lenovo/Videos/Samplevdo.mp4`

 Implementation
- Video input configured in PipeGen.
- Pipeline generated and compiled.
- Person detection and bounding boxes: pending execution verification.
- Tracking, restricted-zone entry, dwell-time alerts, and incident logging: include results only after testing.

 How to Run
Use the `run.sh` script provided in the exported PipeGen project. Pass the input video using the command-line option supported by that script.

 Results
Add screenshots of successful pipeline execution and actual benchmark measurements here.

Limitations
Features that were not implemented or tested are identified as limitations rather than confirmed functionality
