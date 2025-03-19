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
    def __init__(self, part, target_sr=16000, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
            
        self._data_dir = data_dir
        self.target_sr = target_sr
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
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_mix_index.json"
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
        original_index_path = self._data_dir / f"{part}_index.json"
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

        # Create directories for mixes and sources
        mix_dir = self._data_dir / "mix" / part
        mix_dir.mkdir(parents=True, exist_ok=True)
        s1_dir = mix_dir / "s1"
        s2_dir = mix_dir / "s2"
        mix_dir_s = mix_dir / "mix"
        s1_dir.mkdir(exist_ok=True)
        s2_dir.mkdir(exist_ok=True)
        mix_dir_s.mkdir(exist_ok=True)

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

            # Load audio
            waveform1, sr1 = torchaudio.load(entry1["path"])
            waveform2, sr2 = torchaudio.load(entry2["path"])

            # Check sample rate
            if sr1 != sr2:
                continue

            # Trim to same length
            min_len = min(waveform1.size(1), waveform2.size(1))
            waveform1 = waveform1[:, :min_len]
            waveform2 = waveform2[:, :min_len]

            # Create mix
            mix_waveform = (waveform1 + waveform2) / 2

            # Generate filenames
            base_name1 = Path(entry1["path"]).stem
            base_name2 = Path(entry2["path"]).stem
            mix_name = f"{base_name1}_{base_name2}_mix.wav"
            s1_name = f"{base_name1}_s1.wav"
            s2_name = f"{base_name2}_s2.wav"

            # Save files
            mix_path = mix_dir_s / mix_name
            s1_path = s1_dir / s1_name
            s2_path = s2_dir / s2_name

            torchaudio.save(str(mix_path), mix_waveform, sr1)
            torchaudio.save(str(s1_path), waveform1, sr1)
            torchaudio.save(str(s2_path), waveform2, sr1)

            # Add to mix index
            mix_index.append({
                "mix_path": str(mix_path),
                "s1_path": str(s1_path),
                "s2_path": str(s2_path),
                "audio_len": min_len / sr1
            })

        return mix_index
    
    def _get_index_mapping(self, index):
        index_mapping = {}
        cur_index = 0
        for item in index:
            for path in ["s1_path", "s2_path"]:
                speaker_id = Path(path).name.split('-')[0]
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
        mix_audio, sr = self.load_audio(data_dict["mix_path"])
        if sr != self.target_sr:
            mix_audio = torchaudio.functional.resample(mix_audio, sr, self.target_sr)

        audio_list = []
        index_list = []
        for path in ["s1_path", "s2_path"]:
            audio, sr = self.load_audio(data_dict[path])
            if sr != self.target_sr:
                audio = torchaudio.functional.resample(audio, sr, self.target_sr)

            audio_list.append(audio)
            speaker_id = Path(path).name.split('-')[0]
            speaker_index = self.index_mapping[speaker_id] if speaker_id in self.index_mapping else -1
            index_list.append(speaker_index)
            
        '''
        s1_audio, sr = self.load_audio(data_dict["s1_path"])
        if sr != self.target_sr:
            s1_audio = torchaudio.functional.resample(s1_audio, sr, self.target_sr)

        s2_audio, sr = self.load_audio(data_dict["s2_path"])
        if sr != self.target_sr:
            s2_audio = torchaudio.functional.resample(s2_audio, sr, self.target_sr)'''

        audio_len = mix_audio.size(1)
            
        # mix_audio = torch.from_numpy(normalize(mix_audio.numpy(), axis=1) * 0.95)
        # noisy_audio = torch.from_numpy(normalize(noisy_audio.numpy(), axis=1) * 0.95)
        # file_name = data_dict["file_name"]
        # print(file_name)

        instance_data = {"mix_audio": mix_audio,
                         "audios": torch.cat(audio_list),
                         "ids": torch.tensor(index_list, dtype=torch.int32),
                         "audio_len": audio_len,
                         "sr": self.target_sr}
        
        instance_data = self.preprocess_data(instance_data)
        return instance_data

