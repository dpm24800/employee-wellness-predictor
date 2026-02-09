import sys
from src.logger2 import logging


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._error_message_detail(
            error_message, error_detail
        )

    def _error_message_detail(self, error, error_details):
        _, _, exe_tb = error_details.exc_info()
        file_name = exe_tb.tb_frame.f_code.co_filename
        return (
            f"Error occurred in python script: [{file_name}] "
            f"line number: [{exe_tb.tb_lineno}] "
            f"Error Message: [{error}]"
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        1 / 0
    except Exception as e:
        logging.exception("Divide by zero error")
        raise CustomException(e, sys)
