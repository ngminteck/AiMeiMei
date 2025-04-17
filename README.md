# AiMeiMei: AI-Powered Photography Assistant Suite

![AIMeiMei](image.png)

AiMeiMei is a two-part intelligent photo assistant toolkit designed to enhance both the image capturing and image editing experience using advanced AI models. Built with Python and optimized for GPU environments, this suite is ideal for photography learners, tourists, and content creators seeking smart feedback and editing tools.

## AiMeiMei Camera – Smart Photo Quality Scorer
- Capture better photos in real-time with intelligent analysis and feedback.
- SPAQ model for evaluating brightness, contrast, color, noise, and sharpness.
- YOLOv8 for object detection and subject positioning.
- OpenCV-based angle calculation for assessing photo tilt and distortion.
- Real-time suggestions like “Re-center subject” or “Adjust brightness”.
- Individual sub-scores and a composite score to guide the photographer.

## AiMeiMei Photo Editor – Intelligent Image Enhancer
- Edit and enhance images with precision and automation using AI.
- U²-Net and SAM for human segmentation.
- ControlNet and LaMa for AI-based inpainting.
- Real-ESRGAN for 4K upscaling and texture restoration.
- Pillogram for Image Filter.
- SPAQ + YOLO integration for scoring photo quality post-edit.
- Built with PyQt for an intuitive desktop editing experience.

## Credit
Liu MoHan – Team Leader, Assistant Developer & Assistant Report Writer
Led team coordination and planning. Proposed the main idea behind the photo quality scoring system, including the use of YOLO for subject detection and positioning evaluation. Also contributed to YOLO-based implementation and testing, and supported report writing.

Ng Min Teck – Lead Developer
Led the development of the full system (~75% of the codebase), including the implementation of all major features: model integration (SPAQ, YOLO, SAM, LaMa, ControlNet), real-time feedback system, PyQt-based interface, aesthetic and realism scoring, and image editing pipeline. Responsible for feature design, coding, optimization, and system integration.

Tang Liqi – Lead Report Writer & Assistant Developer
Wrote the main report and documentation. Assisted with the AI inpainting modules and helped evaluate the effectiveness of image restoration features.

Zhang Zhiyuan – Assistant Developer & Assistant Report Writer
Collaborated on the photo quality scoring logic alongside Liu MoHan. Contributed to editing and formatting the report, and supported testing and debugging.

Zhu Xiaoyan – Data Collection & PowerPoint
Managed self collection of datasets and resources for testing and evaluation. Designed and prepared presentation slides and visual materials for project demonstration.



