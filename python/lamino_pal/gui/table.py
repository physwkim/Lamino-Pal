"""A QTableWidget filled again whole."""

from __future__ import annotations

from silx.gui import qt


def fill(table: qt.QTableWidget, rows: list[list[qt.QTableWidgetItem]]) -> None:
    """`table` holding `rows`, sorted as it was, its rows selected and current
    before selected and current again by their first cell's text: an index
    names another row once rows are added, removed or sorted."""
    sel = table.selectionModel()
    keys = {table.item(i.row(), 0).text() for i in sel.selectedRows()}
    current = table.item(table.currentRow(), 0).text() if table.currentRow() >= 0 else None
    sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    sel.clear()
    table.setRowCount(len(rows))
    for r, items in enumerate(rows):
        for c, item in enumerate(items):
            table.setItem(r, c, item)
    table.setSortingEnabled(sorting)
    for r in range(table.rowCount()):
        key, index = table.item(r, 0).text(), table.model().index(r, 0)
        if key in keys:
            sel.select(index, qt.QItemSelectionModel.Select | qt.QItemSelectionModel.Rows)
        if key == current:
            sel.setCurrentIndex(index, qt.QItemSelectionModel.NoUpdate)
