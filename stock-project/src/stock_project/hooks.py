from kedro.framework.hooks import hook_impl
import logging

logger = logging.getLogger(__name__)

class Hooks:
    @hook_impl
    def before_node_run(self, node, inputs, catalog, **kwargs):
            
