from shipment.logger import logging
import sys
from shipment.exception import CustomException
from shipment.pipeline.training_pipeline import TrainPipeline


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()