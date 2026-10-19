"""DM0.5 inference support. Import main_dm05 to register the Engine plugin."""

from phyai.models.dm05.configuration_dm05 import DM05Config
from phyai.models.dm05.scheduler_dm05 import DM05Request

__all__ = ["DM05Config", "DM05Request"]
