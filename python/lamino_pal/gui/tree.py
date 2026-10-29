"""The project tree: group, scan, then the volumes made from the scan; h5
files no scan claims and the artifacts below."""

from __future__ import annotations

from typing import Iterator

from silx.gui import qt

from lamino_pal.ops.volume import provenance
from lamino_pal.project import Project

KIND, VALUE = qt.Qt.UserRole, qt.Qt.UserRole + 1


def volumes(project: Project) -> dict[str, list[str]]:
    """Each h5 in the work directory, by the scan tag its provenance names
    ("" for none); paths relative to the work directory."""
    out: dict[str, list[str]] = {}
    for path in sorted(project.work_dir.glob("*.h5")):
        try:
            tag = str(provenance(path).get("tag", ""))
        except OSError:
            continue  # being written
        out.setdefault(tag, []).append(path.name)
    return out


class ProjectTree(qt.QObject):
    """Fills `tree` from the project; `selected` carries (kind, value):
    ("group", name), ("scan", tag), ("volume", path) or ("artifact", path)."""

    selected = qt.Signal(str, str)

    def __init__(self, tree: qt.QTreeWidget, project: Project):
        super().__init__(tree)
        self.tree = tree
        self.project = project
        tree.itemSelectionChanged.connect(self._on_select)

    def refresh(self) -> None:
        keep = self.current()
        self.tree.blockSignals(True)
        self.tree.clear()
        data = self.project.data
        vols = volumes(self.project)
        groups = [(g.name, g.name) for g in data.group]
        if any(s.group is None for s in data.scan):
            groups.append(("(no group)", None))
        for label, name in groups:
            g = self._item(self.tree, label, ("group", name) if name else None)
            for s in data.scan:
                if s.group != name:
                    continue
                si = self._item(g, f"{s.tag}  ({s.kind}, bin {s.bin})", ("scan", s.tag))
                for v in vols.pop(s.tag, []):
                    self._item(si, v, ("volume", v))
        other = [v for vs in vols.values() for v in vs]
        if other:
            o = self._item(self.tree, "Other volumes", None)
            for v in other:
                self._item(o, v, ("volume", v))
        arts = sorted((self.project.work_dir / "artifacts").glob("*.h5"))
        if arts:
            a = self._item(self.tree, "Artifacts", None)
            for path in arts:
                rel = str(path.relative_to(self.project.work_dir))
                self._item(a, path.name, ("artifact", rel))
        self.tree.expandToDepth(1)
        if keep is not None:
            self._select(keep)
        self.tree.blockSignals(False)

    def current(self) -> tuple[str, str] | None:
        items = self.tree.selectedItems()
        return _key(items[0]) if items else None

    def keys(self, kind: str) -> list[str]:
        """The values of the tree's `kind` items, in tree order."""
        return [k[1] for k in map(_key, self._items()) if k is not None and k[0] == kind]

    def pick(self, key: tuple[str, str]) -> None:
        """Select `key`'s item alone and emit `selected`, as a click does,
        also when it was selected already."""
        item = next(i for i in self._items() if _key(i) == key)
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        self.selected.emit(*key)

    def _select(self, key: tuple[str, str]) -> None:
        item = next((i for i in self._items() if _key(i) == key), None)
        if item is not None:
            item.setSelected(True)

    def _items(self, parent: qt.QTreeWidgetItem | None = None) -> Iterator[qt.QTreeWidgetItem]:
        parent = self.tree.invisibleRootItem() if parent is None else parent
        for i in range(parent.childCount()):
            yield parent.child(i)
            yield from self._items(parent.child(i))

    @staticmethod
    def _item(parent, label: str, key: tuple[str, str] | None) -> qt.QTreeWidgetItem:
        item = qt.QTreeWidgetItem(parent, [label])
        if key is not None:
            item.setData(0, KIND, key[0])
            item.setData(0, VALUE, key[1])
        return item

    def _on_select(self) -> None:
        key = self.current()
        if key is not None:
            self.selected.emit(*key)


def _key(item: qt.QTreeWidgetItem) -> tuple[str, str] | None:
    kind = item.data(0, KIND)
    return None if kind is None else (kind, item.data(0, VALUE))
