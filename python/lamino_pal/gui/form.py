"""A form over an op's params: a check box for a bool, a list for a Literal,
text for the rest, where empty text leaves the op's default, or is None for a
required param that takes None."""

from __future__ import annotations

import json
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from silx.gui import qt

from lamino_pal.ops.registry import Op


def _bare(ann: Any) -> Any:
    """`ann` without Annotated and without a None arm."""
    if get_origin(ann) is Annotated:
        return _bare(get_args(ann)[0])
    if get_origin(ann) in (Union, types.UnionType):
        arms = [a for a in get_args(ann) if a is not type(None)]
        if len(arms) == 1:
            return _bare(arms[0])
    return ann


def _nullable(ann: Any) -> bool:
    if get_origin(ann) is Annotated:
        return _nullable(get_args(ann)[0])
    return get_origin(ann) in (Union, types.UnionType) and type(None) in get_args(ann)


class ParamsForm(qt.QWidget):
    def __init__(self, op: Op, parent: qt.QWidget | None = None):
        super().__init__(parent)
        self.op = op
        self._fields: dict[str, tuple[qt.QWidget, Any]] = {}
        self._none: set[str] = set()  # required, and empty means None
        layout = qt.QFormLayout(self)
        for k, f in op.params.model_fields.items():
            ann = _bare(f.annotation)
            default = None if f.is_required() else f.get_default(call_default_factory=True)
            if ann is bool:
                w = qt.QCheckBox()
                w.setChecked(bool(default))
            elif get_origin(ann) is Literal:
                w = qt.QComboBox()
                if not f.is_required():
                    w.addItem("")
                w.addItems([str(a) for a in get_args(ann)])
            else:
                w = qt.QLineEdit()
                if f.is_required() and _nullable(f.annotation):
                    w.setPlaceholderText("required; empty: none")
                    self._none.add(k)
                elif f.is_required():
                    w.setPlaceholderText("required")
                elif default is None:
                    w.setPlaceholderText("auto")
                else:
                    w.setPlaceholderText(default if isinstance(default, str) else json.dumps(default))
            w.setToolTip(str(f.annotation))
            self._fields[k] = (w, ann)
            layout.addRow(k, w)

    def values(self) -> dict[str, Any]:
        """The params set here; text that is not JSON passes as text, for
        the op's validation to judge."""
        out: dict[str, Any] = {}
        for k, (w, ann) in self._fields.items():
            if isinstance(w, qt.QCheckBox):
                out[k] = w.isChecked()
            elif isinstance(w, qt.QComboBox):
                if w.currentText():
                    out[k] = w.currentText()
            elif w.text().strip():
                text = w.text().strip()
                if ann is str:
                    out[k] = text
                else:
                    try:
                        out[k] = json.loads(text)
                    except ValueError:
                        out[k] = text
            elif k in self._none:
                out[k] = None
        return out

    def set(self, **values: Any) -> None:
        """Fill the fields named in `values` that this op has."""
        for k, v in values.items():
            if k not in self._fields:
                continue
            w, ann = self._fields[k]
            if isinstance(w, qt.QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, qt.QComboBox):
                w.setCurrentText("" if v is None else str(v))
            else:
                w.setText("" if v is None else v if isinstance(v, str) else json.dumps(v))
