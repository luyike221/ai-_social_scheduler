"""事件监听器 - 监听各类事件并触发处理"""


class EventListener:
    """事件监听器，负责监听和分发事件"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, event_type: str, handler):
        """注册事件处理器"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def emit(self, event_type: str, event_data: dict):
        """触发事件"""
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event_data)

