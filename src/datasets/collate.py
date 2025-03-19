import torch

from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["mix_audio"] = pad_sequence([x["mix_audio"].transpose(0, 1) for x in dataset_items], batch_first=True).transpose(1, 2)
    result_batch["audios"] = pad_sequence([x["audios"].transpose(0, 1) for x in dataset_items], batch_first=True).transpose(1, 2)
    result_batch["ids"] = torch.stack([x["ids"] for x in dataset_items])
    # result_batch["file_name"] = [x["file_name"] for x in dataset_items]
    result_batch["sr"] = dataset_items[0]["sr"]

    return result_batch

'''
def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["clean_audio"] = pad_sequence([x["clean_audio"].transpose(0, 1) for x in dataset_items], batch_first=True).transpose(1, 2)
    result_batch["noisy_audio"] = pad_sequence([x["noisy_audio"].transpose(0, 1) for x in dataset_items], batch_first=True).transpose(1, 2)
    result_batch["file_name"] = [x["file_name"] for x in dataset_items]
    result_batch["sr"] = dataset_items[0]["sr"]

    return result_batch'''
