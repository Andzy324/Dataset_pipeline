import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
# from dataset_toolkits.utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')
    # label-based targeted download
    parser.add_argument('--label_json', type=str, default=None,
                        help='Path to JSON mapping sha256 -> { "label": [..] } for targeted download')
    parser.add_argument('--labels', type=str, default=None,
                        help='Comma-separated target labels for filtering; match mode controlled by --label_match_mode')
    parser.add_argument('--label_match_mode', type=str, default='any', choices=['any','all'],
                        help='Match mode for labels: any (intersection non-empty) or all (superset)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, **kwargs):    
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]

    # optional: label-based filtering using provided JSON and labels
    label_json_path: Optional[str] = kwargs.get('label_json')
    labels_str: Optional[str] = kwargs.get('labels')
    match_mode: str = kwargs.get('label_match_mode', 'any') or 'any'

    if label_json_path and labels_str:
        import json
        with open(label_json_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        # normalize targets
        target_labels = [s.strip() for s in labels_str.split(',') if s.strip()]
        target_set = set(target_labels)

        # build sha256 -> labels set
        def get_labels_for_sha(sha: str):
            entry = label_map.get(sha) or {}
            labs = entry.get('label') or []
            if isinstance(labs, str):
                labs = [labs]
            return set([str(x).strip() for x in labs if str(x).strip()])

        # filter metadata by label condition
        def match_labels(sha: str) -> bool:
            item_set = get_labels_for_sha(sha)
            if not item_set:
                return False
            if match_mode == 'all':
                return target_set.issubset(item_set)
            # default: any
            return len(target_set & item_set) > 0

        # keep only rows whose sha256 satisfies label match
        if 'sha256' in metadata.columns:
            metadata = metadata[metadata['sha256'].apply(match_labels)]
        else:
            # fallback: if sha256 is index
            metadata = metadata[metadata.index.to_series().apply(match_labels)]
        # sync annotations to filtered sha set
        keep_shas = set(metadata['sha256'].values if 'sha256' in metadata.columns else metadata.index.tolist())
        annotations = annotations[annotations['sha256'].isin(list(keep_shas))]
    
    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
    )
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        sha256 = metadata.loc[k, "sha256"]
        downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    if local_path.startswith('raw/github/repos/'):
                        path_parts = local_path.split('/')
                        file_name = os.path.join(*path_parts[5:])
                        zip_file = os.path.join(output_dir, *path_parts[:5])
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            file = os.path.join(tmp_dir, file_name)
                            record = func(file, sha256)
                    else:
                        file = os.path.join(output_dir, local_path)
                        record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
