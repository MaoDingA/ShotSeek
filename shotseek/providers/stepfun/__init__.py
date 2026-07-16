"""StepFun API adapters used by the M0 contract probe."""

DEFAULT_FILES_BASE_URL = "https://api.stepfun.com/v1"
DEFAULT_CHAT_BASE_URL = "https://api.stepfun.com/step_plan/v1"
DEFAULT_ASR_BASE_URL = "https://api.stepfun.com/v1"
# Backwards-compatible alias for provider-level callers that still import it.
DEFAULT_BASE_URL = DEFAULT_FILES_BASE_URL
DEFAULT_VISION_MODEL = "step-3.7-flash"
DEFAULT_ASR_MODEL = "stepaudio-2.5-asr"
