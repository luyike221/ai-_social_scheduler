"""策略管理器 - 管理运营策略和热点库"""


class StrategyManager:
    """策略管理器，负责管理运营策略和内容模板"""
    
    def __init__(self):
        self.strategies = {}
        self.hot_topics = []
        self.templates = {}
    
    def load_strategy(self, strategy_id: str) -> dict:
        """加载运营策略"""
        return self.strategies.get(strategy_id, {})
    
    def update_hot_topics(self, topics: list):
        """更新热点话题库"""
        self.hot_topics = topics
    
    def get_template(self, template_type: str) -> dict:
        """获取内容模板"""
        return self.templates.get(template_type, {})

