import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
import json
if __name__ == '__main__':


    # python dataset_toolkits/download.py 3D-FUTURE --output_dir datasets/3D-FUTURE --world_size 100
    # python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_sketchfab



    # python dataset_toolkits/download.py HSSD --output_dir datasets/HSSD --world_size 100
    # python dataset_toolkits/build_metadata.py HSSD  --output_dir datasets/HSSD
    # python dataset_toolkits/build_metadata.py 3D-FUTURE --output_dir datasets/3D-FUTURE
    # python dataset_toolkits/download.py ObjaverseXL --output_dir datasets_filtered/ObjaverseXL_sketchfab --world_size 1


    # python dataset_toolkits/download.py ObjaverseXL --output_dir datasets_cano_subset/chair --world_size 1
    # python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets_cano_subset/chair
    # python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets/ObjaverseXL_github


    # python dataset_toolkits/download.py ObjaverseXL --output_dir datasets_cano_subset/monster --world_size 1
    # python dataset_toolkits/build_metadata.py ObjaverseXL --output_dir datasets_cano_subset/monster



    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None, help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(opt.output_dir, exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'local_path' in metadata.columns:
            metadata = metadata[metadata['local_path'].isna()]

        # ── NEW: require a non-empty captions field ────────────────────────────
        if 'captions' in metadata.columns:
            metadata = metadata[
                metadata['captions'].notna() &            # not NaN
                (metadata['captions'].str.strip() != '')  # not just whitespace
            ]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]


    # captions

    filter_ed = False # True
    if filter_ed :
        with open("datasets_filtered/gpt4o_statue.json", "r") as f: # gpt4o_houseboat # gpt4o_armor # statue_(sculpture)
            data_meta = json.load(f)
        print(data_meta.keys())
        print("data_meta.keys() length:",len(data_meta.keys()))
        # # save keys
        # out_path = "datasets/objaverse_lvis/annotation_keys.txt"
        # with open(out_path, "w") as fout:
        #     for k in data_meta.keys():
        #         fout.write(f"{k}\n")

        # print(f"Saved {len(data_meta.keys())} keys to {out_path}")
        cls_name= 'statue_(sculpture)' # houseboat # armor # statue_(sculpture)
        chair_list = data_meta[cls_name]
        print("chair_list length:",len(chair_list))
        shape_ids = set(chair_list)
 
        metadata = metadata[metadata['file_identifier'].apply(lambda x: str(x).split('/')[-1] in shape_ids)]



    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]


    # metadata=metadata.head(100)  
    print(f'Processing {len(metadata)} objects...')

    # exit(0)
    # process objects
    downloaded = dataset_utils.download(metadata, **opt)
    downloaded.to_csv(os.path.join(opt.output_dir, f'downloaded_{opt.rank}.csv'), index=False)


    # download some shapes categorized by gpt4o
    # python dataset_toolkits/download.py ObjaverseXL --output_dir datasets_filtered/ObjaverseXL_sketchfab --world_size 1