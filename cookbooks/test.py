import warnings
from peft import PeftModel
import torch
from transformers import AutoProcessor
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
import random
import numpy as np
from loguru import logger
from sentence_manager.sentence_manager import SentenceNormalizer
import re
import yaml

def seed_everything(seed=1895):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
warnings.filterwarnings("ignore")

class MingAudio:
    def __init__(self, model_path, lora_path=None, device="cuda:0", use_grouped_gemm=False):
        self.device = device
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        if use_grouped_gemm and not self.model.config.llm_config.use_grouped_gemm:
            self.model.model.fuse_experts()

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model = self.model.eval().to(torch.bfloat16).to(self.device)
        self.processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.sample_rate = self.processor.audio_processor.sample_rate
        self.patch_size = self.processor.audio_processor.patch_size
        self.normalizer = self.init_tn_normalizer(tokenizer=self.tokenizer)

    def init_tn_normalizer(self, config_file_path=None, tokenizer=None):
        if config_file_path is None:
            default_config_path = "sentence_manager/default_config.yaml"
            config_file_path = default_config_path
        
        with open(config_file_path, 'r') as f:
            self.sentence_manager_config = yaml.safe_load(f)
        
        if "split_token" not in self.sentence_manager_config:
            self.sentence_manager_config["split_token"] = []
        
        assert isinstance(self.sentence_manager_config["split_token"], list)
        if tokenizer is not None:
            self.sentence_manager_config["split_token"].append(re.escape(tokenizer.eos_token))

        normalizer = SentenceNormalizer(self.sentence_manager_config.get("text_norm", {}))
        
        return normalizer

    def speech_understanding(self, messages, lang=None):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.device)
        
        if lang is not None:
            language = torch.tensor([self.tokenizer.encode(f'{lang}\t')]).to(inputs['input_ids'].device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], language], dim=1)
            attention_mask = inputs['attention_mask']
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=attention_mask.dtype)
        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        logger.info(f"input: {self.tokenizer.decode(inputs['input_ids'].cpu().numpy().tolist()[0])}")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=self.processor.gen_terminator,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def speech_generation(
        self, 
        text,
        prompt_wav_path,
        prompt_text,
        lang='zh',
        output_wav_path='out.wav'
    ):
        text = self.normalizer.normalize(text)
        waveform = self.model.generate_tts(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            patch_size=self.patch_size,
            tokenizer=self.tokenizer,
            lang=lang,
            output_wav_path=output_wav_path,
            sample_rate=self.sample_rate,
            device=self.device
        )
        
        return waveform

    def speech_edit(
        self, 
        messages,
        output_wav_path='out.wav',
        use_cot=True
    ):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.device)

        if use_cot:
            ans = torch.tensor([self.tokenizer.encode('<answer>')]).to(inputs['input_ids'].device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], ans], dim=1)
            attention_mask = inputs['attention_mask']
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=attention_mask.dtype)
        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        logger.info(f"input: {self.tokenizer.decode(inputs['input_ids'].cpu().numpy().tolist()[0])}")

        edited_speech, edited_text = self.model.generate_edit(
            **inputs,
            tokenizer=self.tokenizer,
            output_wav_path=output_wav_path
        )
        return edited_speech, edited_text

if __name__ == "__main__":
    model = MingAudio("inclusionAI/Ming-UniAudio-16B-A3B")
    # ASR
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "Please recognize the language of this speech and transcribe it. Format: oral.",
                },
                
                {"type": "audio", "audio": "data/wavs/BAC009S0915W0292.wav"},
            ],
        },
    ]
    
    response = model.speech_understanding(messages=messages)
    logger.info(f"Generated Response: {response}")

    # TTS
    model.speech_generation(
        text='我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。',
        prompt_wav_path='data/wavs/10002287-00000094.wav',
        prompt_text='在此奉劝大家别乱打美白针。',
        output_wav_path='data/output/tts.wav'
    )

    # Edit
    del model
    model = MingAudio("inclusionAI/Ming-UniAudio-16B-A3B-Edit")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": "data/wavs/00004768-00000024.wav", "target_sample_rate": 16000},
                {
                    "type": "text",
                    "text": "<prompt>Please recognize the language of this speech and transcribe it. And insert '实现' before the character or word at index 3.\n</prompt>",
                },
            ],
        },
    ]
    
    response = model.speech_edit(messages=messages, output_wav_path='data/output/ins.wav')
    logger.info(f"Generated Response: {response}")
