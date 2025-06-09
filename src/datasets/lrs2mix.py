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


FOLDER_NAMES = {
    "train": "tr",
    "val": "cv",
    "test": "tt"
}


class LRS2Mix(BaseDataset):
    def __init__(self, part, target_sr=16000, segment_size=None, data_dir=None, index_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "lrs2mix"
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
        
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load(self):
        arch_path = self._index_dir / "lrs2.tar.gz"    
        print(f"Loading data")
        wget.download("https://huggingface.co/datasets/JusperLee/LRS2-2Mix/resolve/main/lrs2.tar.gz", str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "lrs2_rebuild" / "audio" / "wav16k" / "min").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "lrs2_rebuild"))


    def _get_or_load_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            if not (self._data_dir / FOLDER_NAMES[split]).exists():
                self._load()
            index = []
            for fpath in tqdm((self._data_dir / FOLDER_NAMES[split] / "mix").iterdir()):
                index.append(
                    {
                        "s1_path": str(self._data_dir / FOLDER_NAMES[split] / "s1" / fpath.name),
                        "s2_path": str(self._data_dir / FOLDER_NAMES[split] / "s2" / fpath.name),
                        "mix_path": str(self._data_dir / FOLDER_NAMES[split] / "mix" / fpath.name),
                    }
                )
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
    
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
        mix_audio, sr_mix = self.load_audio(data_dict["mix_path"])

        if sr1 != self.target_sr:
            s1_audio = torchaudio.functional.resample(s1_audio, sr1, self.target_sr)
        if sr2 != self.target_sr:
            s2_audio = torchaudio.functional.resample(s2_audio, sr2, self.target_sr)
        if sr_mix != self.target_sr:
            mix_audio = torchaudio.functional.resample(mix_audio, sr_mix, self.target_sr)

        min_len = min(s1_audio.size(1), s2_audio.size(1))
        s1_audio = s1_audio[:, :min_len]
        s2_audio = s2_audio[:, :min_len]

        audio_start = 0

        audio_len = mix_audio.size(1)
        if self.segment_size is not None and self.segment_size < audio_len:
            audio_start = 0 # random chunk?
            audio_len = self.segment_size
            mix_audio = mix_audio[:, audio_start : audio_start + audio_len]

        audio_list = []
        index_list = []
        for audio, _ in [(s1_audio, data_dict["s1_path"]), (s2_audio, data_dict["s2_path"])]:
            audio = audio[:, audio_start : audio_start + audio_len]

            audio_list.append(audio)
            index_list.append(-1)

        instance_data = {"mix_audio": mix_audio,
                         "audios": torch.cat(audio_list),
                         "ids": torch.tensor(index_list, dtype=torch.int32),
                         "audio_len": audio_len,
                         "sr": self.target_sr}
        
        instance_data = self.preprocess_data(instance_data)
        return instance_data
