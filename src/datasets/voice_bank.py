import json
import re
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from librosa.util import normalize
from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
import random
import json
import os
import shutil
from pathlib import Path
import wget


URL_LINKS = {
    "train": {
        "clean": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
        "noisy": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"
    },
    "test": {
        "clean": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
        "noisy": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
    },
    "train-clean": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
    "train-noisy": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip",
    "test-clean": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
    "test-noisy": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
}

URL_LINK = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/"

FOLDER_NAMES = {
    "train": ["clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"],
    "test": ["clean_testset_wav", "noisy_testset_wav"]
}


class VoiceBankDataset(BaseDataset):
    def __init__(self, split="train", target_sr=16000, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "voice_bank"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self.target_sr = target_sr

        index = self._get_or_load_index(split)
        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        for subfolder_name in FOLDER_NAMES[part]:
            arch_path = self._data_dir / f"{part}/{subfolder_name}.zip"
            (self._data_dir / f"{part}").mkdir(exist_ok=True, parents=True)
            print(f"Loading part {part}")
            wget.download(f"{URL_LINK}{subfolder_name}.zip", str(arch_path))
            shutil.unpack_archive(arch_path, self._data_dir)
            os.remove(str(arch_path))

    def _get_or_load_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            if not (self._data_dir / FOLDER_NAMES[split][0]).exists():
                self._load_part(split)
            index = []
            for fpath in tqdm((self._data_dir / FOLDER_NAMES[split][0]).iterdir()):
                index.append(
                    {
                        "noisy_path": str(self._data_dir / FOLDER_NAMES[split][1] / fpath.name),
                        "clean_path": str(self._data_dir / FOLDER_NAMES[split][0] / fpath.name),
                        "file_name": str(fpath.name)
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
        noisy_audio, sr = self.load_audio(data_dict["noisy_path"])
        if sr != self.target_sr:
            noisy_audio = torchaudio.functional.resample(noisy_audio, sr, self.target_sr)

        clean_audio, sr = self.load_audio(data_dict["clean_path"])
        if sr != self.target_sr:
            clean_audio = torchaudio.functional.resample(clean_audio, sr, self.target_sr)

        audio_len = noisy_audio.size(1)
        split = 32768
        if audio_len > split:
            max_audio_start = audio_len - split
            audio_start = random.randint(0, max_audio_start)
            audio_len = split
            noisy_audio = noisy_audio[:, audio_start : audio_start + split]
            clean_audio = clean_audio[:, audio_start : audio_start + split]
            
        clean_audio = torch.from_numpy(normalize(clean_audio.numpy(), axis=1) * 0.95)
        noisy_audio = torch.from_numpy(normalize(noisy_audio.numpy(), axis=1) * 0.95)
        file_name = data_dict["file_name"]

        instance_data = {"noisy_audio": noisy_audio,
                         "clean_audio": clean_audio,
                         "audio_len": audio_len,
                         "file_name": file_name,
                         "sr": self.target_sr}
        
        instance_data = self.preprocess_data(instance_data)
        return instance_data
