from typing import List, Tuple, Dict, Optional, Any, Union
import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.utils import TensorType
from transformers.feature_extraction_utils import FeatureExtractionMixin, BatchFeature

NORM_FACTOR_FOR_DTYPE = {
    torch.int8: 2**7,
    torch.int16: 2**15,
    torch.int32: 2**31,
    torch.int64: 2**63,
    torch.float32: 1,
    torch.float64: 1,
}

# special tokens
DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"
DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>"
DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>"
DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"
DEFAULT_TEXT_TOKEN = '<text>'
DEFAULT_ASR_TOKEN = '<asr>'
DEFAULT_TTS_TOKEN = '<tts>'


class BailingMMAudioProcessor(FeatureExtractionMixin):
    def __init__(self, audio_tokenizer_args: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 16000
        self.hop_size = audio_tokenizer_args["hop_size"]
        self.patch_size = audio_tokenizer_args["patch_size"]
        self.sampling_rate = audio_tokenizer_args["sampling_rate"]

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output

    def __call__(self, audios, **kwargs) -> BatchFeature:
        """Preprocess an audio or a batch of audios."""
        return self.preprocess(audios, **kwargs)

    def _make_batched_waveforms(self, waveform_list):
        waveform_length = torch.tensor([x.size(-1) for x in waveform_list], dtype=torch.int)
        waveform  = pad_sequence(waveform_list, batch_first=True, padding_value=0)
        encoder_feats_lengths = torch.ceil(waveform_length / self.hop_size)
        encoder_feats_lengths = encoder_feats_lengths // self.patch_size
        return {"waveform": waveform, "waveform_length": waveform_length, "encoder_feats_lengths": encoder_feats_lengths}
        
    def preprocess(
        self,
        audios: Union[Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(audios, List):
            audio_inputs = self._make_batched_waveforms(
                [waveform[0] for waveform, sr in audios]
            )
        else:
            waveform, sr = audios
            assert waveform.ndim == 2
            audio_inputs = self._make_batched_waveforms([waveform[0]])
        return BatchFeature(data=audio_inputs, tensor_type=return_tensors)
