from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy

from invokeai.app.invocations.primitives import BoardField, ColorField, VideoField, VideoOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.backend.image_util.invisible_watermark import InvisibleWatermark
from invokeai.backend.image_util.safety_checker import SafetyChecker

from .baseinvocation import (
    BaseInvocation,
    Classification,
    Input,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)

@invocation("show_video", title="Show Video", tags=["video"], category="video", version="1.0.0")
class ShowVideoInvocation(BaseInvocation):
    """Displays a provided video using the OS video viewer, and passes it forward in the pipeline."""

    video_field: VideoField = InputField(description="The video to show")

    def invoke(self, context: InvocationContext) -> VideoOutput:
        video = Path(self.video_field.video_name)

        if not video.exists():
            raise RuntimeError(f"Video {video} does not exist")

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"video [{video}] could not be opened")

        width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)

        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                cv2.imshow(str(video), frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        capture.release()
        cv2.destroyWindow(str(video))

        return VideoOutput(
                video=VideoField(video_name=str(video)),
                width=int(width),
                height=int(height),
                fps=fps,
                frames=frames
                )
