import json
import re
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

import json
import os
import shutil
from pathlib import Path
import wget
import time


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


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
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

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index


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
            '''
            print(arch_path, self._data_dir)
            for fpath in (self._data_dir / subfolder_name).iterdir():
                shutil.move(str(fpath), str(self._data_dir / fpath.name))
            os.remove(str(arch_path))
            shutil.rmtree(str(self._data_dir / subfolder_name))'''
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
                        "noisy_path": str(fpath),
                        "clean_path": str(self._data_dir / FOLDER_NAMES[split][1] / fpath.name),
                        "file_name": str(fpath.name)
                    }
                )
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
            '''
            index = []
            print(len(self._dataset))
            print(self._dataset[0])
            for i, entry in tqdm(enumerate(self._dataset)):
                # print(entry)

                entry["index"] = i
                # entry["noisy_path"] = str(Path(entry['noisy']["path"]).absolute().resolve())
                # entry["text"] = self._regex.sub("", entry.get("sentence", "").lower())
                # t_info = torchaudio.info(entry["clean_path"])
                # entry["audio_len"] = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = entry["clean"]["array"].size
                # print(entry["audio_len"])
                index.append(
                    {
                        "index": i,
                        "audio_len": entry["audio_len"],
                    }
                )
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)'''
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
        file_name = data_dict["file_name"]

        instance_data = {"noisy_audio": noisy_audio,
                         "clean_audio": clean_audio,
                         "audio_len": audio_len,
                         "file_name": file_name,
                         "sr": self.target_sr}
        
        instance_data = self.preprocess_data(instance_data)
        return instance_data
