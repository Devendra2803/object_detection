from sqlalchemy import Column, Integer, String, DateTime, Text, LargeBinary
from db import Base
from datetime import datetime

class DetectionModelTraining(Base):
    __tablename__ = "detection_model_training"

    id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String, nullable=False)
    task_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_path = Column(String)  # ✅ Add this line
    raw_data_zip = Column(String, nullable=False)
    num_classes = Column(Integer)
    class_names = Column(Text)
    data_size = Column(Integer)
    splitted_data = Column(String)
    video_output = Column(LargeBinary)
    status = Column(String, default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)
