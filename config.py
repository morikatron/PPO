from typing import NamedTuple


class Config(NamedTuple):
    # learning config
    env_name: str = "CartPole-v1"  # OpenAI gymの環境名
    seed: int = 1234  # tensorflow numpy randomのシード値
    num_update: int = 5000  # 学習時の総イテレーション数
    log_step: int = 100  # ログを表示する頻度
    play: bool = False  # 学習終了時に描画して再生
    # hyper-parameters
    batch_size: int = 32  # バッチサイズ
    num_epoch: int = 4  # エポック数
    num_step: int = 128  # 1イテレーションで保存するサンプルの総数(horizon)
    num_unit: int = 64  # fc層のunit数
    gamma: float = 0.99  # 割引率
    lambda_: float = 0.95  # GAEの割引率
    clip: float = 0.2  # クリップする範囲
    vf_coef: float = 0.5  # 価値損失の係数
    ent_coef: float = 0.01  # エントロピー項の係数
    learning_rate: float = 2.5e-4  # 学習率
    gradient_clip: float = 0.5  # 勾配をクリップする範囲
