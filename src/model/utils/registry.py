from typing import Any, Dict, Optional, Callable
from collections.abc import MutableMapping
import copy


def convert_dict_key_to_lower_case(dictionary: MutableMapping) -> Dict:
    out = {}
    for key, value in dictionary.items():
        if isinstance(value, MutableMapping):
            value = convert_dict_key_to_lower_case(value)
        out[key.lower()] = value
    return out


class Registry():
    __OBJ_DICT__ = {}

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Any] = dict()
        self.__OBJ_DICT__[name] = self

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    @property
    def name(self):
        return self._name

    def get(self, name: str) -> Any:
        module = self._module_dict.get(name, None)
        if module is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry"
                + f"Registry: {set(self._module_dict.keys())}"
            )
        return module

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._module_dict
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._module_dict[name] = obj

    def register(self, module: Optional[Any] = None) -> Any:
        if module is None:
            def deco(_module: Any) -> Any:
                name = _module.__name__
                self._do_register(name, _module)
                return _module
            return deco
        else:
            name = module.__name__
            self._do_register(name, module)
            return module

    @classmethod
    def build_from_config(
        cls,
        config: MutableMapping,
        registry_key: str,
        module_key: str,
        convert_key_to_lower_case: bool = True,
        safe_build: bool = True
    ):
        _config = config
        _registry_key = registry_key
        _module_key = module_key
        if convert_key_to_lower_case:
            config = convert_dict_key_to_lower_case(_config)
            registry_key = _registry_key.lower()
            module_key = _module_key.lower()
        elif safe_build:
            config = copy.deepcopy(_config)
        else:
            config = _config
        assert registry_key in config, f'registry key ({_registry_key}) not in config (keys: {set(_config.keys())})'
        assert module_key in config, f'module key ({_module_key}) not in config ({set(_config.keys())})'
        return cls._do_build_from_config(config, registry_key, module_key)

    @classmethod
    def _do_build_from_config(
        cls,
        config: MutableMapping,
        registry_key: str,
        module_key: str,
    ):
        registry_name = config.pop(registry_key)
        module_name = config.pop(module_key)
        registry = cls.__OBJ_DICT__[registry_name]
        module = registry.get(module_name)

        for key, value in config.items():
            if isinstance(value, MutableMapping) and registry_key in value and module_key in value:
                config[key] = cls._do_build_from_config(value, registry_key, module_key)
        return module(**config)
