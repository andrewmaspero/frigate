from typing import Optional

from pydantic import Field

from frigate.const import AUDIO_MIN_CONFIDENCE

from ..base import FrigateBaseModel

__all__ = ["AudioConfig", "AudioFilterConfig", "ExternalAudioConfig"]


DEFAULT_LISTEN_AUDIO = ["bark", "fire_alarm", "scream", "speech", "yell"]


class AudioFilterConfig(FrigateBaseModel):
    threshold: float = Field(
        default=0.8,
        ge=AUDIO_MIN_CONFIDENCE,
        lt=1.0,
        title="Minimum detection confidence threshold for audio to be counted.",
    )


class ExternalAudioConfig(FrigateBaseModel):
    """Configuration for external audio detection."""
    enabled: bool = Field(default=False, title="Enable external audio detection")
    endpoint: str = Field(
        default="ipc:///tmp/cache/zmq_audio", title="ZMQ IPC endpoint for audio detection"
    )
    request_timeout_ms: int = Field(
        default=200, title="ZMQ request timeout in milliseconds"
    )
    linger_ms: int = Field(default=0, title="ZMQ socket linger in milliseconds")


class AudioConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enable audio events.")
    max_not_heard: int = Field(
        default=30, title="Seconds of not hearing the type of audio to end the event."
    )
    min_volume: int = Field(
        default=500, title="Min volume required to run audio detection."
    )
    listen: list[str] = Field(
        default=DEFAULT_LISTEN_AUDIO, title="Audio to listen for."
    )
    filters: Optional[dict[str, AudioFilterConfig]] = Field(
        None, title="Audio filters."
    )
    enabled_in_config: Optional[bool] = Field(
        None, title="Keep track of original state of audio detection."
    )
    num_threads: int = Field(default=2, title="Number of detection threads", ge=1)
    external: Optional[ExternalAudioConfig] = Field(
        default_factory=ExternalAudioConfig, title="External audio detection configuration"
    )
