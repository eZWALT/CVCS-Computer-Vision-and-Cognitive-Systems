import torch
import torch.nn as nn

class VisionConnector(nn.Module):
    """
    Maps vision encoder features -> LLM embedding space
    Wide + shallow MLP (recommended)
    """
    def __init__(self, vision_dim, llm_dim, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, vision_feats):
        # vision_feats: (B, N, vision_dim)
        return self.proj(vision_feats)  # (B, N, llm_dim)


class SimpleVLM(nn.Module):
    """
    LLaVA-style VLM:
    - Frozen vision encoder
    - Frozen LLM
    - Trainable connector
    """
    def __init__(self, vision_encoder, llm, connector):
        super().__init__()
        self.vision = vision_encoder
        self.llm = llm
        self.connector = connector

        # Freeze vision + LLM
        self.vision.requires_grad_(False)
        self.llm.requires_grad_(False)

    def forward(
        self,
        images,                 # preprocessed images
        input_ids,              # text token IDs (with <image>)
        image_token_id,         # int
        labels=None
    ):
        """
        Returns LLM outputs (loss + logits)
        """

        # ---- vision ----
        with torch.no_grad():
            vision_feats = self.vision(images).last_hidden_state
            # (B, N, vision_dim)

        # ---- connector ----
        visual_tokens = self.connector(vision_feats)
        # (B, N, llm_dim)

        # ---- text embeddings ----
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # (B, T, llm_dim)

        # ---- replace <image> token with visual tokens ----
        B = input_ids.size(0)
        full_embeds = []

        for b in range(B):
            idx = (input_ids[b] == image_token_id).nonzero()[0].item()
            merged = torch.cat([
                text_embeds[b, :idx],
                visual_tokens[b],
                text_embeds[b, idx+1:]
            ], dim=0)
            full_embeds.append(merged)

        full_embeds = torch.stack(full_embeds, dim=0)

        # ---- LLM ----
        outputs = self.llm(
            inputs_embeds=full_embeds,
            labels=labels
        )

        return outputs
