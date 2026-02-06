from util.import_util import script_imports

script_imports()

import json
from enum import Enum
from typing import Any, get_args, get_origin

from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.type_util import issubclass_safe


def _jsonable_default(value: Any) -> Any:
    if issubclass_safe(type(value), BaseConfig):
        return None
    if issubclass_safe(type(value), Enum):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [(_jsonable_default(v)) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[str(k)] = _jsonable_default(v)
        return out
    return str(value)


def _kind_of(var_type: type) -> str:
    if issubclass_safe(var_type, BaseConfig):
        return "object"
    if issubclass_safe(var_type, Enum):
        return "enum"
    if var_type is bool:
        return "bool"
    if var_type is int:
        return "int"
    if var_type is float:
        return "float"
    if var_type is str:
        return "str"
    return "json"


def _flatten_config(base: BaseConfig, prefix: str = "") -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for name in base.types:
        var_type = base.types[name]
        nullable = bool(base.nullables.get(name, False))
        default = base.default_values.get(name)
        path = f"{prefix}.{name}" if prefix else name

        if issubclass_safe(var_type, BaseConfig):
            child = getattr(base, name)
            if issubclass_safe(type(child), BaseConfig):
                fields.extend(_flatten_config(child, prefix=path))
            continue

        kind = _kind_of(var_type)
        enum_values = None
        item_kind = None

        if var_type is list or get_origin(var_type) is list:
            kind = "list"
            args = get_args(var_type)
            if args:
                item_kind = _kind_of(args[0])
        elif var_type is dict or get_origin(var_type) is dict:
            kind = "dict"
        elif issubclass_safe(var_type, Enum):
            enum_values = [e.name for e in var_type]

        fields.append(
            {
                "path": path,
                "kind": kind,
                "nullable": nullable,
                "default": _jsonable_default(default),
                "enum": enum_values,
                "item_kind": item_kind,
            }
        )

    return fields


def main():
    cfg = TrainConfig.default_values()
    fields = _flatten_config(cfg)
    out = {
        "ok": True,
        "schema_version": 1,
        "fields": fields,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

