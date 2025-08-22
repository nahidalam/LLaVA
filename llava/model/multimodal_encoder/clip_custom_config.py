from transformers.models.clip.configuration_clip import CLIPVisionConfig

class CLIPVisionCustomConfig(CLIPVisionConfig):
    model_type = CLIPVisionConfig.model_type  # keep same type

    def __init__(
        self,
        use_vision_rope_2d: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        **kwargs,
    ):
        # take custom fields from kwargs if they came via from_pretrained
        use_vision_rope_2d = kwargs.pop("use_vision_rope_2d", use_vision_rope_2d)
        rope_theta = kwargs.pop("rope_theta", rope_theta)

        # let the base class handle all standard fields (and any extra like _name_or_path)
        super().__init__(**kwargs)

        # set custom fields
        self.use_vision_rope_2d = bool(use_vision_rope_2d)
        self.rope_theta = float(rope_theta)
