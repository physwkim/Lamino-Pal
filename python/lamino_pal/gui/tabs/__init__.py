"""The tabs, in the order the guide takes the steps."""

from lamino_pal.gui.tabs.align import AlignTab
from lamino_pal.gui.tabs.batch import BatchTab
from lamino_pal.gui.tabs.data import DataTab
from lamino_pal.gui.tabs.export import ExportTab
from lamino_pal.gui.tabs.preview import PreviewTab
from lamino_pal.gui.tabs.raw import RawTab
from lamino_pal.gui.tabs.volume import VolumeTab

TABS = (DataTab, RawTab, AlignTab, PreviewTab, BatchTab, VolumeTab, ExportTab)
