import sys
from src.logger import logging

def error_message_detail(error, error_details: sys):
    _, _, exe_tb = error_details.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename

    # error_message = "Error occured in python script: [{0}] line number: [{1}] " \
    # "Error Message: [{2}]".format( file_name, exe_tb.tb_lineno, str(error))

    error_message = (
        "Error occurred in python script: "
        "[{0}] line number: [{1}] Error Message: [{2}]"
    ).format(
        file_name,
        exe_tb.tb_lineno,
        str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_details=error_detail)

    def __str__(self):
        return self.error_message

# Only execute this code when this file is run directly, not when imported.
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)