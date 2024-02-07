from typing import Any, Dict, Optional, Tuple

class trainer_base():
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.2),
        image_size: int = 512,
        num_class: int = 2,
        class_name: list = ["NRG", "RG"],
        kfold_seed: int = 111,
        kfold_index: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        is_transform=True,
        balance_data=True,
        binary_unbalance_train_ratio=100
    ) -> None:

    def epoch_loop(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def test(self):
        pass
