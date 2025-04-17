import json
import os
import shutil
from pathlib import Path

import torch
import torchaudio
from librosa.util import normalize
import wget
from tqdm import tqdm
import random

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechMixDataset(BaseDataset):
    def __init__(self, part, target_sr=16000, segment_size=None, data_dir=None, index_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)

        if index_dir is None:
            index_dir = data_dir
        else:
            index_dir = Path(index_dir)
            index_dir.mkdir(exist_ok=True, parents=True)
            
        self._data_dir = data_dir
        self._index_dir = index_dir
        self.target_sr = target_sr
        self.segment_size = segment_size
        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        self.index_mapping = self._get_index_mapping(index)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._index_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_mix_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            original_index = self._get_or_load_original_index(part)
            index = self._create_mix_index(original_index, part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _get_or_load_original_index(self, part):
        original_index_path = self._index_dir / f"{part}_index.json"
        if original_index_path.exists():
            with original_index_path.open() as f:
                original_index = json.load(f)
        else:
            original_index = self._create_original_index(part)
            with original_index_path.open("w") as f:
                json.dump(original_index, f, indent=2)
        return original_index

    def _create_original_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any(f.endswith(".flac") for f in filenames):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "audio_len": length,
                        }
                    )
        return index

    def _create_mix_index(self, original_index, part):
        mix_index = []
        used_pairs = set()

        # Shuffle the original index to randomize pairs
        random.seed(42)
        shuffled_index = original_index.copy()
        random.shuffle(shuffled_index)

        # Generate mixes
        for i in tqdm(range(len(shuffled_index)), desc=f"Creating mixes for {part}"):
            j = (i + 1) % len(shuffled_index)
            entry1 = shuffled_index[i]
            entry2 = shuffled_index[j]

            # Avoid mixing same file
            if entry1["path"] == entry2["path"]:
                continue

            # Check for duplicate pairs
            pair_key = tuple(sorted([entry1["path"], entry2["path"]]))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            # Add to mix index
            mix_index.append({
                "s1_path": entry1["path"],
                "s2_path": entry2["path"],
            })

        return mix_index
    
    def _get_index_mapping(self, index):
        index_mapping = {}
        cur_index = 0
        for item in index:
            for path in ["s1_path", "s2_path"]:
                speaker_id = Path(item[path]).name.split('-')[0]
                if speaker_id not in index_mapping:
                    index_mapping[speaker_id] = cur_index
                    cur_index += 1
        return index_mapping
    
    def __getitem__(self, ind):
        """
        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        s1_audio, sr1 = self.load_audio(data_dict["s1_path"])
        s2_audio, sr2 = self.load_audio(data_dict["s2_path"])

        if sr1 != self.target_sr:
            s1_audio = torchaudio.functional.resample(s1_audio, sr1, self.target_sr)
        if sr2 != self.target_sr:
            s2_audio = torchaudio.functional.resample(s2_audio, sr2, self.target_sr)

        min_len = min(s1_audio.size(1), s2_audio.size(1))
        s1_audio = s1_audio[:, :min_len]
        s2_audio = s2_audio[:, :min_len]

        mix_audio = (s1_audio + s2_audio) / 2

        audio_start = 0

        audio_len = mix_audio.size(1)
        if self.segment_size is not None and self.segment_size < audio_len:
            audio_start = 0 # random chunk?
            audio_len = self.segment_size
            mix_audio = mix_audio[:, audio_start : audio_start + audio_len]

        audio_list = []
        index_list = []
        for audio, audio_path in [(s1_audio, data_dict["s1_path"]), (s2_audio, data_dict["s2_path"])]:
            audio = audio[:, audio_start : audio_start + audio_len]

            audio_list.append(audio)
            speaker_id = Path(audio_path).name.split('-')[0]
            speaker_index = self.index_mapping[speaker_id] if speaker_id in self.index_mapping else -1
            index_list.append(speaker_index)

        instance_data = {"mix_audio": mix_audio,
                         "audios": torch.cat(audio_list),
                         "ids": torch.tensor(index_list, dtype=torch.int32),
                         "audio_len": audio_len,
                         "sr": self.target_sr}
        
        instance_data = self.preprocess_data(instance_data)
        return instance_data
