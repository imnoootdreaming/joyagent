"""SQLAlchemy Base —— 独立文件，避免循环导入。"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """所有模型的声明式基类。"""
    pass
