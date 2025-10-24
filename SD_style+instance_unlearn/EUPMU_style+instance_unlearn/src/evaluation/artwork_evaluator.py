import json
import os
from argparse import ArgumentParser

import pandas as pd
from prettytable import PrettyTable

from src.configs.generation_config import GenerationConfig

from .eval_util import clip_score, clip_eval_by_image
from .evaluator import Evaluator, GenerationDataset

ARTWORK_DATASETS = {
    "art": "benchmark/art_prompts.csv",
    "artwork": "benchmark/artwork_prompts.csv",
    "big_artists": "/media/toby/SSD2/linux_workspaces/machine_unlearning/SPM/benchmark/big_artists_prompts.csv",
    "famous_art": "benchmark/famous_art_prompts.csv",
    "generic_artists": "benchmark/generic_artists_prompts.csv",
    "kelly": "benchmark/kelly_prompts.csv",
    "niche_art": "benchmark/niche_art_prompts.csv",
    "short_niche_art": "benchmark/short_niche_art_prompts.csv",
    "short_vangogh": "benchmark/short_vangogh_prompts.csv",
    "vangogh": "benchmark/vangogh_prompts.csv",
    "picasso": "benchmark/picasso_prompts.csv",
    "rembrandt": "benchmark/rembrandt_prompts.csv",
    "andy_warhol": "benchmark/andy_warhol_prompts.csv",
    "caravaggio": "benchmark/caravaggio_prompts.csv",
}


class ArtworkDataset(GenerationDataset):
    def __init__(
        self,
        datasets: list[str],
        save_folder: str = "benchmark/generated_imgs/",
        base_cfg: GenerationConfig = GenerationConfig(),
        num_images_per_prompt: int = 20,
        **kwargs,
    ) -> None:
        assert all([dataset in ARTWORK_DATASETS for dataset in datasets]), (
            f"datasets should be a subset of {ARTWORK_DATASETS}, " f"got {datasets}."
        )

        meta = {}
        self.data = []
        for dataset in datasets:
            meta[dataset] = {}
            df = pd.read_csv(ARTWORK_DATASETS[dataset])
            for idx, row in df.iterrows():
                cfg = base_cfg.copy()
                cfg.prompts = [row["prompt"]]
                cfg.seed = row["evaluation_seed"]
                cfg.generate_num = num_images_per_prompt
                cfg.save_path = os.path.join(
                    save_folder,
                    dataset,
                    f"{idx}" + "_{}.png",
                )
                self.data.append(cfg.dict())
                meta[dataset][row["prompt"]] = [
                    cfg.save_path.format(i) for i in range(num_images_per_prompt)
                ]
        os.makedirs(save_folder, exist_ok=True)
        meta_path = os.path.join(save_folder, "meta.json")
        print(f"Saving metadata to {meta_path} ...")
        with open(meta_path, "w") as f:
            json.dump(meta, f)


class ArtworkEvaluator(Evaluator):
    """
    Evaluation for artwork on CLIP-protocol accepts `save_folder` as a *JSON file* with the following format:
    {
        DATASET_1: {
            PROMPT_1_1: [IMAGE_PATH_1_1_1, IMAGE_PATH_1_1_2, ...],
            PROMPT_1_2: [IMAGE_PATH_1_2_1, IMAGE_PATH_1_2_2, ...],
            ...
        },
        DATASET_2: {
            PROMPT_2_1: [IMAGE_PATH_2_1_1, IMAGE_PATH_2_1_2, ...],
            PROMPT_2_2: [IMAGE_PATH_2_2_1, IMAGE_PATH_2_2_2, ...],
            ...
        },
        ...
    }
    DATASET_i: str, the i-th concept to be evaluated.
    PROMPT_i_j: int, the j-th prompt in DATASET_i.
    IMAGE_PATH_i_j_k: str, the k-th image path for DATASET_i, PROMPT_i_j.
    """

    def __init__(
        self,
        save_folder: str = "benchmark/generated_imgs/",
        output_path: str = "benchmark/results/",
        eval_with_template: bool = False,
    ):
        super().__init__(save_folder=save_folder, output_path=output_path)
        self.img_metadata = json.load(open(os.path.join(self.save_folder, "meta.json")))
        self.eval_with_template = eval_with_template

    def evaluation(self):
        scores_dict = {}
        accs_dict = {}
        scores = 0.0
        accs = 0.0
        for dataset, data in self.img_metadata.items():
            score = 0.0
            num_images = 0
            for prompt, img_paths in data.items():
                clip_prompt = [prompt] * len(img_paths) if self.eval_with_template else [dataset.replace("_", " ")] * len(img_paths)
                #print(f"Evaluating dataset: {dataset}, prompt: {prompt}, clip_prompt: {clip_prompt}")
                #assert False
                #clip_prompt = ["painting"] * len(img_paths)
                #cs = clip_score(
                #    img_paths,
                #    clip_prompt
                #)
                #score += cs.mean().item() * len(img_paths)
                score, acc = clip_eval_by_image(
                    img_paths,
                    clip_prompt,
                    ["painting"] * len(img_paths),
                )
                scores += score * len(img_paths)
                accs += acc * len(img_paths)
                num_images += len(img_paths)
            scores /= num_images
            accs /= num_images
            scores_dict[dataset] = scores
            accs_dict[dataset] = accs

                #num_images += len(img_paths)
                #print(f"Dataset: {dataset}, Prompt: {prompt}, Score: {cs}")
            #scores[dataset] = score / num_images

        #table = PrettyTable()
        #table.field_names = ["Dataset", "CLIPScore"]
        #for dataset, score in scores.items():
        #    table.add_row([dataset, score])
        #print(table)
        #with open(os.path.join(self.output_path, "scores.json"), "w") as f:
        #    json.dump(scores, f)
        table = PrettyTable()
        table.field_names = ["Concept", "CLIPScore", "CLIPAccuracy"]

        for dataset, score in scores_dict.items():
            table.add_row([dataset, scores_dict[dataset], accs_dict[dataset]])
        print(table)

        save_name = "evaluation_results.json" if self.eval_with_template else "evaluation_results(concept only).json"
        with open(os.path.join(self.output_path, save_name), "w") as f:
            json.dump([scores_dict, accs_dict], f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--save_folder",
        type=str,
        help="path to json that contains metadata for generated images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to save evaluation results.",
    )
    args = parser.parse_args()

    evaluator = ArtworkEvaluator(
        save_folder=args.save_folder, output_path=args.output_path
    )
    evaluator.evaluation()
