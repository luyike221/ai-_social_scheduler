"""状态管理器 - 跟踪任务状态和系统状态"""


class StateManager:
    """状态管理器，负责跟踪任务和系统状态"""
    
    def __init__(self):
        self.task_states = {}
        self.session_states = {}
        self.platform_states = {}
    
    def update_task_state(self, task_id: str, state: dict):
        """更新任务状态"""
        self.task_states[task_id] = state
    
    def get_task_state(self, task_id: str) -> dict:
        """获取任务状态"""
        return self.task_states.get(task_id, {})
    
    def save_session(self, session_id: str, context: dict):
        """保存会话状态"""
        self.session_states[session_id] = context

