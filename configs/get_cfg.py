import os
import yaml

from configs.config import YAML_PATH
from configs.schema import Config, CameraConfig


class ConfigError(Exception):
    """설정 로드/검증 실패 시 사용."""


def _check_exists(path, name):
    if not path:
        raise ConfigError(f"{name} 경로가 비어 있습니다.")
    if not os.path.exists(path):
        raise ConfigError(f"{name} 경로가 존재하지 않습니다: {path}")
    return path


def get_cfg():
    if not os.path.exists(YAML_PATH):
        raise ConfigError(f"CONFIG_PATH가 가리키는 파일을 찾을 수 없습니다: {YAML_PATH}")

    with open(YAML_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not config or not isinstance(config, dict):
        raise ConfigError("설정 파일이 비어 있거나 형식이 올바르지 않습니다.")

    # 필수 키 검증
    try:
        model = config['MODEL']
        label = config['LABEL']
        delegate = config.get('DELEGATE')
        _check_exists(model, "MODEL")
        _check_exists(label, "LABEL")
        if delegate:
            _check_exists(delegate, "DELEGATE")

        cam = config['CAMERA']
        _ = cam['IR']
        _ = cam['RGB_FRONT']
    except KeyError as e:
        raise ConfigError(f"필수 설정 키가 없습니다: {e}") from e

    return _to_dataclass(config)


def _to_dataclass(raw: dict) -> Config:
    """dict 설정을 dataclass Config로 변환"""
    cam = raw['CAMERA']
    return Config(
        MODEL=raw['MODEL'],
        LABEL=raw['LABEL'],
        DELEGATE=raw.get('DELEGATE', ""),
        CAMERA_IR=CameraConfig(**cam['IR']),
        CAMERA_RGB_FRONT=CameraConfig(**cam['RGB_FRONT']),
        TARGET_RES=tuple(raw.get('TARGET_RES', (0, 0))),
        SERVER=raw.get('SERVER', {}),
        DISPLAY=raw.get('DISPLAY', {}),
        SYNC=raw.get('SYNC', {}),
        INPUT=raw.get('INPUT', {}),
        STATE=raw.get('STATE', {}),
        CAPTURE=raw.get('CAPTURE', {}),
        COORD=raw.get('COORD', {}),
    )
