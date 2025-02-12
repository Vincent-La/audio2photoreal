'''
Perform end to end inference: audio --> photorealistic avatar

NOTE: video generation requires ffmpeg to be available

'''
import argparse
import os
import copy
import json
from typing import Dict, Union

# import gradio as gr
import numpy as np
import torch
import torchaudio
from attrdict import AttrDict
from diffusion.respace import SpacedDiffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from model.diffusion import FiLMTransformer
from utils.misc import fixseed
from utils.model_util import create_model_and_diffusion, load_model
from visualize.render_codes import BodyRenderer


# Mapping of person_id to checkpoint folder names
ID_DICT = {
    'PXB184': 'c1',
    'RLW104': 'c2',
    'TXB805': 'c3',
    'GQS883': 'c4'
}

# finds model checkpoint in path (should only be one .pt file)
def find_pt(path):
    return [file for file in os.listdir(path) if file.endswith('.pt')][0]

# Adapted from demo.GradioModel
class A2PModel:
    def __init__(self, person_id, output_dir):
        
        self.person_id = person_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print('Output dir: ', self.output_dir)

        self.name = ID_DICT[person_id]

        # model args
        face_args= f"checkpoints/diffusion/{self.name}_face/args.json"
        print('face_args', face_args)
        pose_args= f"checkpoints/diffusion/{self.name}_pose/args.json"
        print('pose_args: ', pose_args)

        # model checkpoints
        face_ckpt = f"checkpoints/diffusion/{self.name}_face"
        face_ckpt = os.path.join(face_ckpt, find_pt(face_ckpt))
        print('face_ckpt', face_ckpt)

        pose_ckpt = f"checkpoints/diffusion/{self.name}_pose"
        pose_ckpt = os.path.join(pose_ckpt, find_pt(pose_ckpt))
        print('pose_ckpt', pose_ckpt)


        # face model setup
        self.face_model, self.face_diffusion, self.device = self._setup_model(
            face_args, face_ckpt
        )

        # pose model setup
        self.pose_model, self.pose_diffusion, _ = self._setup_model(
            pose_args, pose_ckpt
        )

        # load stats for standardization
        data_stats_path = f'dataset/{person_id}/data_stats.pth'
        stats = torch.load(data_stats_path)
        stats["pose_mean"] = stats["pose_mean"].reshape(-1)
        stats["pose_std"] = stats["pose_std"].reshape(-1)
        self.stats = stats

        # set up renderer
        config_base = f"checkpoints/ca_body/data/{person_id}"
        self.body_renderer = BodyRenderer(
            config_base=config_base,
            render_rgb=True,
        )

    
    def _setup_model(
        self,
        args_path: str,
        model_path: str,
    ) -> (Union[FiLMTransformer, ClassifierFreeSampleModel], SpacedDiffusion):
        
        with open(args_path) as f:
            args = json.load(f)

        args = AttrDict(args)
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("running on...", args.device)
        args.model_path = model_path
        # args.output_dir = "/tmp/gradio/"
        args.output_dir = self.output_dir
        args.timestep_respacing = "ddim100"
        if args.data_format == "pose":
            args.resume_trans = f"checkpoints/guide/{self.name}_pose/checkpoints"
            args.resume_trans = os.path.join(args.resume_trans, find_pt(args.resume_trans))

            print('guide checkpoint:', args.resume_trans)

        ## create model
        model, diffusion = create_model_and_diffusion(args, split_type="test")
        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location=args.device)
        load_model(model, state_dict)
        model = ClassifierFreeSampleModel(model)
        model.eval()
        model.to(args.device)
        return model, diffusion, args.device
    

    def _replace_keyframes(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        B: int,
        T: int,
        top_p: float = 0.97,
    ) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.pose_model.transformer.generate(
                model_kwargs["y"]["audio"],
                T,
                layers=self.pose_model.tokenizer.residual_depth,
                n_sequences=B,
                top_p=top_p,
            )
        tokens = tokens.reshape((B, -1, self.pose_model.tokenizer.residual_depth))
        pred = self.pose_model.tokenizer.decode(tokens).detach()
        return pred

    def _run_single_diffusion(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        diffusion: SpacedDiffusion,
        model: Union[FiLMTransformer, ClassifierFreeSampleModel],
        curr_seq_length: int,
        num_repetitions: int = 1,
    ) -> (torch.Tensor,):
        sample_fn = diffusion.ddim_sample_loop
        with torch.no_grad():
            sample = sample_fn(
                model,
                (num_repetitions, model.nfeats, 1, curr_seq_length),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        return sample

    def generate_sequences(
        self,
        model_kwargs: Dict[str, Dict[str, torch.Tensor]],
        data_format: str,
        curr_seq_length: int,
        num_repetitions: int = 5,
        guidance_param: float = 10.0,
        top_p: float = 0.97,
        # batch_size: int = 1,
    ) -> Dict[str, np.ndarray]:
        if data_format == "pose":
            model = self.pose_model
            diffusion = self.pose_diffusion
        else:
            model = self.face_model
            diffusion = self.face_diffusion

        all_motions = []
        model_kwargs["y"]["scale"] = torch.ones(num_repetitions) * guidance_param
        model_kwargs["y"] = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        if data_format == "pose":
            model_kwargs["y"]["mask"] = (
                torch.ones((num_repetitions, 1, 1, curr_seq_length))
                .to(self.device)
                .bool()
            )
            model_kwargs["y"]["keyframes"] = self._replace_keyframes(
                model_kwargs,
                num_repetitions,
                int(curr_seq_length / 30),
                top_p=top_p,
            )
        sample = self._run_single_diffusion(
            model_kwargs, diffusion, model, curr_seq_length, num_repetitions
        )
        all_motions.append(sample.cpu().numpy())
        print(f"created {len(all_motions) * num_repetitions} samples")
        return np.concatenate(all_motions, axis=0)


def generate_results(a2p_model:A2PModel, audio: np.ndarray, num_repetitions: int, top_p: float):
    # if audio is None:
    #     raise gr.Error("Please record audio to start")

    sr, y = audio
    # set to mono and perform resampling
    y = torch.Tensor(y)
    if y.dim() == 2:
        dim = 0 if y.shape[0] == 2 else 1
        y = torch.mean(y, dim=dim)
    y = torchaudio.functional.resample(torch.Tensor(y), orig_freq=sr, new_freq=48_000)
    sr = 48_000

    # NOTE: this part crops audio to 4 seconds for the demo
    # # make it so that it is 4 seconds long
    # if len(y) < (sr * 4):
    #     raise gr.Error("Please record at least 4 second of audio")
    # if num_repetitions is None or num_repetitions <= 0 or num_repetitions > 10:
    #     raise gr.Error(
    #         f"Invalid number of samples: {num_repetitions}. Please specify a number between 1-10"
    #     )
    # cutoff = int(len(y) / (sr * 4))
    # y = y[: cutoff * sr * 4]
    
    curr_seq_length = int(len(y) / sr) * 30
    # create model_kwargs
    model_kwargs = {"y": {}}
    dual_audio = np.random.normal(0, 0.001, (1, len(y), 2))
    dual_audio[:, :, 0] = y / max(y)
    dual_audio = (dual_audio - a2p_model.stats["audio_mean"]) / a2p_model.stats[
        "audio_std_flat"
    ]
    model_kwargs["y"]["audio"] = (
        torch.Tensor(dual_audio).float().tile(num_repetitions, 1, 1)
    )
    face_results = (
        a2p_model.generate_sequences(
            model_kwargs, "face", curr_seq_length, num_repetitions=int(num_repetitions)
        )
        .squeeze(2)
        .transpose(0, 2, 1)
    )
    face_results = (
        face_results * a2p_model.stats["code_std"] + a2p_model.stats["code_mean"]
    )
    pose_results = (
        a2p_model.generate_sequences(
            model_kwargs,
            "pose",
            curr_seq_length,
            num_repetitions=int(num_repetitions),
            guidance_param=2.0,
            top_p=top_p,
        )
        .squeeze(2)
        .transpose(0, 2, 1)
    )
    pose_results = (
        pose_results * a2p_model.stats["pose_std"] + a2p_model.stats["pose_mean"]
    )
    dual_audio = (
        dual_audio * a2p_model.stats["audio_std_flat"]
        + a2p_model.stats["audio_mean"]
    )
    return face_results, pose_results, dual_audio[0].transpose(1, 0).astype(np.float32)


def audio_to_avatar(a2p_model:A2PModel, audio: np.ndarray, num_repetitions: int, top_p: float):
    face_results, pose_results, audio = generate_results(a2p_model, audio, num_repetitions, top_p)
    # returns: num_rep x T x 104
    B = len(face_results)
    results = []
    for i in range(B):
        render_data_block = {
            "audio": audio,  # 2 x T
            "body_motion": pose_results[i, ...],  # T x 104
            "face_motion": face_results[i, ...],  # T x 256
        }
        a2p_model.body_renderer.render_full_video(
            render_data_block, 
            os.path.join(a2p_model.output_dir, f"sample{i}"), 
            audio_sr=48_000
        )
    #     results += [gr.Video(value=f"/tmp/sample{i}_pred.mp4", visible=True)]
    # results += [gr.Video(visible=False) for _ in range(B, 10)]
    # return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--person_id',
        type=str,
        required=True,
        choices= ['PXB184', 'RLW104', 'TXB805', 'GQS883'],
        help = 'Person ID to select face, pose, guide, and avatar models'
    )

    parser.add_argument(
        '--input_audio',
        type = str,
        required=True,
        help = 'Path to audio (.wav) file as input'
    )

    parser.add_argument(
        '--output_dir',
        type = str,
        required=True,
        help='Path to output files to'
    )

    parser.add_argument(
        '--num_samples',
        type = int,
        required=False,
        default=3,
        help = 'Number of samples to generate'
    )

    parser.add_argument(
        '--sample_diversity',
        type = float,
        required=False,
        default=0.97,
        help = 'Tunes the cumulative probability in nucleus sampling: 0.01 = low diversity, 1.0 = high diversity.'
    )

    args = parser.parse_args()
    print(args)

    # load model(s)
    model = A2PModel(args.person_id, args.output_dir)

    # load audio
    y, sr = torchaudio.load(args.input_audio)
    input_audio = (sr,y)

    # some util function that fixes a bunch of seeds for deterministic output
    fixseed(42)

    # INFERENCE
    audio_to_avatar(a2p_model=model, 
                    audio=input_audio,
                    num_repetitions=args.num_samples,
                    top_p=args.sample_diversity)
    
    print('FINISH INFERENCE')

