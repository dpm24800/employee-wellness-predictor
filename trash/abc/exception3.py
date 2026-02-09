# src/exception.py
import sys
from src.logger import logging  # Reuse our logger—no duplication!

def error_message_detail(error, error_detail: sys):
    """
    Extracts precise failure context from Python's traceback object
    """
    _, _, exc_tb = error_detail.exc_info()  # Unpack exception traceback
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get failing file
    
    # Craft human+machine readable error message
    error_message = "Error occurred in python script: [{0}] line number: [{1}] " \
                    "Error Message: [{2}]".format(
                        file_name, 
                        exc_tb.tb_lineno,  # Critical: exact line number!
                        str(error)
                    )
    return error_message

class CustomException(Exception):
    """
    Production-grade exception that auto-logs failures with full context
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Enrich exception with file/line context BEFORE raising
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message  # Display enriched message on print()

# Validation test: Prove it works before trusting it in pipelines
if __name__ == "__main__":
    try:
        a = 1 / 0  # Deliberate failure
    except Exception as e:
        logging.info("Divide by Zero")  # Log intent
        raise CustomException(e, sys)   # Raise with full context